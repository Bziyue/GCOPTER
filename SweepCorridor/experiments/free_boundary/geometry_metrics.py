"""Offline geometric bending metrics, independent of longitudinal speed."""
import numpy as np

def bending(path,order=24,angles=48):
    cp=np.loadtxt(path,delimiter=',',skiprows=1).reshape(-1,4,12)
    h=(cp[:,0,2]-cp[:,0,1])[:,None,None];y=cp[:,:,3:]
    t,w=np.polynomial.legendre.leggauss(order);t=(t+1)/2;w=w/2;r=1-t
    b=np.array([r**3,3*r*r*t,3*r*t*t,t**3]).T
    db=np.array([-3*r*r,3*r*r-6*r*t,6*r*t-3*t*t,3*t*t]).T
    ddb=np.array([6*r,18*t-12,6-18*t,6*t]).T
    value=np.einsum('kj,pjd->pkd',b,y)
    dy=np.einsum('kj,pjd->pkd',db,y)/h;ddy=np.einsum('kj,pjd->pkd',ddb,y)/(h*h)
    theta=2*np.pi*np.arange(angles)/angles;z=np.array([np.cos(theta),np.sin(theta)]);zt=np.array([-np.sin(theta),np.cos(theta)])
    def shape(v,q):return np.einsum('pkij,jt->pkit',v[:,:,3:].reshape(-1,order,2,3).transpose(0,1,3,2),q).transpose(0,1,3,2)
    velocity=dy[:,:,:3][:,:,None,:]+shape(dy,z)
    acceleration=ddy[:,:,:3][:,:,None,:]+shape(ddy,z)
    theta_first=shape(value,zt);mixed=shape(dy,zt);theta_second=-shape(value,z)
    speed=np.linalg.norm(velocity,axis=-1)
    density=np.linalg.norm(np.cross(velocity,acceleration),axis=-1)**2/speed**5
    integrate=lambda v:float(np.sum(v.mean(axis=2)*w[None,:]*h[:,:,0]))
    longitudinal=integrate(density)
    normal=np.cross(velocity,theta_first);area=np.linalg.norm(normal,axis=-1);normal/=area[:,:,:,None]
    dot=lambda a,b:np.sum(a*b,axis=-1)
    e,f,g=dot(normal,acceleration),dot(normal,mixed),dot(normal,theta_second)
    E,F,G=dot(velocity,velocity),dot(velocity,theta_first),dot(theta_first,theta_first)
    determinant=area*area
    # Squared norm of the shape operator: k1^2+k2^2 = 4H^2-2K.
    squared_curvature=((e*G-2*f*F+g*E)/determinant)**2-2*(e*g-f*f)/determinant
    surface=2*np.pi*integrate(np.maximum(0,squared_curvature)*area)
    surface_area=2*np.pi*integrate(area)
    return dict(longitudinal=longitudinal,surface=surface,surface_area=surface_area,mean_surface_curvature=surface/surface_area)
