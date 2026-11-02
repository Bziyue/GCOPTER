#!/usr/bin/env python3
"""Replay RViz input topics against real mockamap + GCOPTER Informed RRT*.
Uses numpy only for an independent voxel-sphere audit; never substitutes an A* seed.
"""
import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import numpy as np
import rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, PointStamped
from nav_msgs.msg import Path as PathMsg
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String, Float64MultiArray, Empty

parser = argparse.ArgumentParser()
parser.add_argument('--scene', choices=['perlin', 'maze'], default='perlin')
parser.add_argument('--output', required=True)
parser.add_argument('--search-timeout', type=float, default=1.)
parser.add_argument('--domain', type=int, default=169)
parser.add_argument('--repeats', type=int, default=1)
args = parser.parse_args()
out = Path(args.output).resolve()
out.mkdir(parents=True, exist_ok=True)
os.environ['ROS_DOMAIN_ID'] = str(args.domain)
os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'] = 'LOCALHOST'
log = (out/'launch.log').open('w')
proc = subprocess.Popen(['ros2', 'launch', 'gcopter', 'direct_corridor.launch.py',
                         'scene:='+args.scene, 'rviz:=false',
                         'search_timeout:='+str(args.search_timeout), 'output_dir:='+str(out)],
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
rclpy.init()
node = rclpy.create_node('direct_demo_regression')
received = {'status': '', 'cloud': None, 'metrics': None, 'controls': [], 'reference': None}
qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
subs = [node.create_subscription(String, '/direct_demo/status', lambda m: received.update(status=m.data), qos),
        node.create_subscription(PointCloud2, '/voxel_map', lambda m: received.update(cloud=m), 1),
        node.create_subscription(String, '/direct_demo/metrics', lambda m: received.update(metrics=json.loads(m.data) if m.data else None), qos),
        node.create_subscription(Float64MultiArray, '/direct_demo/controls', lambda m: received.update(controls=list(m.data)), qos),
        node.create_subscription(PathMsg, '/direct_demo/reference', lambda m: received.update(reference=m), qos)]
goal_pub = node.create_publisher(PoseStamped, '/move_base_simple/goal', 10)
start_pub = node.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
point_pub = node.create_publisher(PointStamped, '/clicked_point', 10)
reset_pub = node.create_publisher(Empty, '/direct_demo/reset', 10)

def wait(predicate, timeout=40):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError('launch exited; inspect '+str(out/'launch.log'))
        rclpy.spin_once(node, timeout_sec=.02)
        if predicate():
            return
    raise RuntimeError('timeout; status='+received['status'])

def spin_for(seconds):
    deadline = time.monotonic()+seconds
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=.02)

def publish_point(p, actual_z=False, frame='odom'):
    if actual_z:
        msg = PointStamped()
        msg.header.frame_id = frame
        msg.point.x, msg.point.y, msg.point.z = map(float, p)
        point_pub.publish(msg)
    else:
        msg = PoseStamped()
        msg.header.frame_id = frame
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, p)
        msg.pose.orientation.w = 1.
        goal_pub.publish(msg)

records = []
try:
    wait(lambda: received['status']=='WAITING_FOR_START' and received['cloud'] is not None)
    cloud = received['cloud']
    extent = 50 if args.scene=='perlin' else 30
    lower = np.array([-extent/2, -extent/2, 0.])
    upper = np.array([extent/2, extent/2, 5.])
    res = .25
    dims = np.ceil((upper-lower)/res).astype(int)
    offsets = {f.name: f.offset for f in cloud.fields}
    dtype = np.dtype({'names':['x','y','z'], 'formats':['<f4']*3,
                      'offsets':[offsets[k] for k in ('x','y','z')], 'itemsize':cloud.point_step})
    raw = np.frombuffer(cloud.data, dtype=dtype)
    xyz = np.column_stack([raw[k] for k in ('x','y','z')]).astype(float)
    ids = np.floor((xyz-lower)/res).astype(int)
    ids = ids[((ids>=0)&(ids<dims)).all(axis=1)]
    occupied = np.zeros(tuple(dims), dtype=bool)
    occupied[tuple(ids.T)] = True
    # Exact distance to all nearby occupied-voxel circumspheres. No generator
    # support-plane data, nearest-path projection or corridor checks are reused.
    def nearest(points, radius):
        points = np.asarray(points)
        cells = np.floor((points-lower)/res).astype(int)
        best = np.full(len(points), np.inf)
        n = math.ceil(radius/res)
        for dx in range(-n,n+1):
            for dy in range(-n,n+1):
                for dz in range(-n,n+1):
                    if sum(max(abs(v)-.5,0)**2 for v in (dx,dy,dz))*res**2 > radius**2:
                        continue
                    idx = cells + [dx,dy,dz]
                    valid = np.flatnonzero(((idx>=0)&(idx<dims)).all(axis=1))
                    hits = valid[occupied[tuple(idx[valid].T)]]
                    if len(hits):
                        dist = np.linalg.norm(points[hits]-(lower+(idx[hits]+.5)*res), axis=1)
                        best[hits] = np.minimum(best[hits],dist)
        return best

    def free_near(anchor):
        samples = np.array([[x,y,anchor[2]]
                            for x in np.arange(lower[0]+.75,upper[0]-.75,.5)
                            for y in np.arange(lower[1]+.75,upper[1]-.75,.5)])
        padding = .2+.05+.2
        mask = ((samples>lower+padding)&(samples<upper-padding)).all(axis=1)
        samples = samples[mask]
        distances = nearest(samples,math.sqrt(3)*res/2+padding+.1)
        samples = samples[distances>math.sqrt(3)*res/2+padding+.04]
        if not len(samples):
            raise RuntimeError('no known-free test endpoint near '+str(anchor))
        return samples[np.argmin(np.linalg.norm(samples-anchor,axis=1))]

    a = extent*.4
    anchors = [([-a,-a,2.5],[a,a,2.5]),([-a,a,2.5],[a,-a,2.5]),
               ([-a,0,1.5],[a,0,3.5])]
    if args.scene == 'maze':
        anchors = [([-a,-a,2.5],[-a,a,2.5]), ([-a,0,2.5],[a,0,2.5]),
                   ([-a,-a,1.5],[a,a,3.5]), ([-a,a,2.5],[a,-a,2.5])]
    pairs = [(free_near(s),free_near(g)) for s,g in anchors]
    expected = ['PASS'] * len(pairs)
    if args.scene == 'maze':
        # This particular original mockamap maze has an isolated room. Prove
        # disconnection on raw occupancy, before accepting SEARCH_FAILED.
        # Flood fill only classifies endpoints; it NEVER supplies a seed path.
        assert (occupied == occupied[:,:,:1]).all(), 'expected full-height walls'
        labels = np.full(tuple(dims[:2]), -1, dtype=int)
        component = 0
        for i,j in zip(*np.where(~occupied[:,:,0])):
            if labels[i,j] >= 0:
                continue
            pending = deque([(i,j)])
            labels[i,j] = component
            while pending:
                x,y = pending.popleft()
                for dx,dy in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
                    nx,ny = x+dx,y+dy
                    if 0<=nx<dims[0] and 0<=ny<dims[1] and not occupied[nx,ny,0] and labels[nx,ny]<0:
                        labels[nx,ny] = component
                        pending.append((nx,ny))
            component += 1
        def component_at(p):
            return labels[tuple(np.floor((p[:2]-lower[:2])/res).astype(int))]
        for case,(s,g) in enumerate(pairs):
            assert (component_at(s)==component_at(g)) == (case!=3), 'unexpected maze connectivity'
        expected[3] = 'SEARCH_FAILED'
    (out/'inputs.json').write_text(json.dumps(dict(scene=args.scene,cloud_sha256=hashlib.sha256(cloud.data).hexdigest(),
        points=int(len(xyz)),occupied_voxels=int(occupied.sum()),expected=expected,
        pairs=[(s.tolist(),g.tolist()) for s,g in pairs]),indent=2))
    last_request = 0
    for repeat in range(args.repeats):
        for case,(start,goal) in enumerate(pairs):
            for reverse in (False,True):
                s,g = (goal,start) if reverse else (start,goal)
                reset_pub.publish(Empty())
                wait(lambda: received['status']=='WAITING_FOR_START' and not received['controls'])
                publish_point(s,actual_z=case==2)
                wait(lambda: received['status']=='WAITING_FOR_GOAL')
                publish_point(g,actual_z=case==2)
                wait(lambda: received['metrics'] is not None and received['metrics']['request']>last_request,
                     timeout=max(20,args.search_timeout+10))
                metric = dict(received['metrics'])
                last_request = metric['request']
                metric.update(case=case,reverse=reverse,repeat=repeat,expected=expected[case])
                if metric['status']=='PASS':
                    wait(lambda: len(received['controls'])==metric['controls']*11 and received['reference'] is not None and len(received['reference'].poses)>0)
                    controls = np.array(received['controls']).reshape(-1,11)
                    samples = []
                    for u in np.linspace(0,1,401):
                        ends = controls[::6,1]
                        span = min(len(ends)-1, int(np.searchsorted(ends,u,side='right')))
                        block = controls[6*span:6*span+6]
                        t = (u-block[0,0])/(block[0,1]-block[0,0])
                        b = np.array([math.comb(5,j)*t**j*(1-t)**(5-j) for j in range(6)])
                        c = b@block[:,2:]
                        shape = c[3:].reshape(2,3).T
                        for radius in (0.,.5,1.):
                            for angle in np.linspace(0,2*np.pi,24,endpoint=False):
                                samples.append(c[:3]+shape@np.array([radius*np.cos(angle),radius*np.sin(angle)]))
                    samples = np.array(samples)
                    clearance = nearest(samples,math.sqrt(3)*res/2+.25+.1)-math.sqrt(3)*res/2-.25
                    assert clearance.min()>-1e-7, 'independent sphere collision'
                    assert ((samples>=lower+.25-1e-8)&(samples<=upper-.25+1e-8)).all(), 'boundary violation'
                    ref = received['reference'].poses
                    first = np.array([ref[0].pose.position.x,ref[0].pose.position.y,ref[0].pose.position.z])
                    last = np.array([ref[-1].pose.position.x,ref[-1].pose.position.y,ref[-1].pose.position.z])
                    assert np.linalg.norm(first-s)<1e-8 and np.linalg.norm(last-g)<1e-8
                    metric['audit_samples'] = len(samples)
                    metric['sampled_min_clearance'] = float(clearance.min())
                    np.savetxt(out/f'controls_{last_request}.csv',controls,delimiter=',')
                    spin_for(1.1)
                    assert received['metrics']['request']==last_request and received['controls'], 'unchanged periodic map erased result'
                else:
                    wait(lambda: not received['controls'])
                records.append(metric)
                (out/'results.json').write_text(json.dumps(records,indent=2))
                print(args.scene,case,'reverse' if reverse else 'forward',metric['status'],
                      'search_ms=',round(metric['search_ms'],2),'corridor_ms=',round(metric['corridor_ms'],2),
                      'turns=',metric['turns_over_30deg'],flush=True)
    # Same interfaces as RViz: invalid goal, explicit initial pose, reset.
    publish_point([1000,1000,2.5])
    wait(lambda: received['status'].startswith('INVALID_POINT') and not received['controls'])
    explicit = PoseWithCovarianceStamped()
    explicit.header.frame_id='odom'
    explicit.pose.pose.position.x,explicit.pose.pose.position.y,explicit.pose.pose.position.z=map(float,pairs[0][0])
    explicit.pose.pose.orientation.w=1.
    start_pub.publish(explicit)
    wait(lambda: received['status']=='WAITING_FOR_GOAL')
    reset_pub.publish(Empty())
    wait(lambda: received['status']=='WAITING_FOR_START' and not received['controls'])
    passed = sum(m['status']=='PASS' for m in records)
    matched = sum(m['status']==m['expected'] for m in records)
    print(f'{passed}/{len(records)} corridors; {matched}/{len(records)} expected outcomes; '
          'invalid selection / explicit start / reset passed',flush=True)
    if matched!=len(records):
        raise SystemExit(1)
finally:
    node.destroy_node()
    rclpy.shutdown()
    if proc.poll() is None:
        os.killpg(proc.pid,signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            proc.wait(timeout=5)
    log.close()
