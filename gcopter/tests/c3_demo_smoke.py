#!/usr/bin/env python3
# ROS integration smoke test: publish the same goal messages as RViz2.
import argparse,json,os,signal,subprocess,time
from pathlib import Path
import rclpy
from rclpy.qos import QoSProfile,DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray
parser=argparse.ArgumentParser();parser.add_argument('--degree',type=int,choices=(3,5),default=3)
degree=parser.parse_args().degree
mode='c3' if degree==3 else 'smooth'
root=Path(__file__).resolve().parents[2]
(root/f'.scratch/sweep_rviz_d{degree}').mkdir(parents=True,exist_ok=True)
inputs=json.loads((root/'docs/validation/native_splines/frozen_manifest.json').read_text())['inputs']
rclpy.init();node=rclpy.create_node('c3_demo_smoke')
q=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
state={}
subs=[node.create_subscription(String,'/direct_demo/status',lambda x:state.update(status=x.data),q),node.create_subscription(String,'/direct_demo/metrics',lambda x:state.update(metrics=json.loads(x.data)) if x.data else state.pop("metrics",None),q),node.create_subscription(MarkerArray,'/direct_demo/corridor',lambda x:state.update(mesh=sum(len(m.points) for m in x.markers)),q)]
pub=node.create_publisher(PoseStamped,'/move_base_simple/goal',10)
def wait(test,seconds=40):
 end=time.monotonic()+seconds
 while time.monotonic()<end:
  rclpy.spin_once(node,timeout_sec=.1)
  if test():return
 raise RuntimeError(str(state))
try:
 for scene,case in [('perlin',0),('maze',2)]:
  state.clear()
  with (root/f'.scratch/sweep_rviz_d{degree}/{scene}.log').open('w') as log:
   process=subprocess.Popen([str(root/'gcopter/tools/run_direct_demo.sh'),f'scene:={scene}','rviz:=false',f'shape_mode:={mode}','search_timeout:=3.0',f'output_dir:={root}/.scratch/sweep_rviz_d{degree}/{scene}'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:
    wait(lambda:state.get('status')=='WAITING_FOR_START')
    state.pop('metrics',None);state.pop('mesh',None)
    lines=(root/inputs[case]['seed']).read_text().splitlines()
    for line in [lines[0],lines[-1]]:
     x,y,z=map(float,line.split());msg=PoseStamped();msg.header.frame_id='odom';msg.pose.position.x=x;msg.pose.position.y=y;msg.pose.position.z=z;msg.pose.orientation.w=1.;pub.publish(msg)
     if line==lines[0]:wait(lambda:state.get('status')=='WAITING_FOR_GOAL')
    wait(lambda:'metrics' in state)
    m=state['metrics'];assert m['status']=='PASS',m
    assert m['bounds_lower']==inputs[case]['bounds_lower']
    wait(lambda:state.get('mesh',0)>0)
    assert m['representation']==('native_cubic_v3' if degree==3 else 'quintic_hermite_v2') and not m['trajectory_optimization']
    assert m['corridor_ms']>0 and state['mesh']>0
    (root/f'.scratch/sweep_rviz_d{degree}/{scene}_smoke.json').write_text(json.dumps(state,indent=2))
    print(scene,m['corridor_ms'],state['mesh'],flush=True)
   finally:
    os.killpg(process.pid,signal.SIGINT);process.wait(timeout=15)
finally:node.destroy_node();rclpy.shutdown()
