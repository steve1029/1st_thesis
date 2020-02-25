from KEMP import structures
import numpy as np

class square_ring(object):
    def __new__(self, mat1, mat2, base_com, a1, a2, b1, b2, thickness, angle=0., rot_axis=(0.,0.,1.)):
        sqr  = structures.Box(mat1, ((base_com[0]-a1/2., base_com[1]-b1/2., base_com[2]), (base_com[0]+a1/2., base_com[1]+b1/2., base_com[2]+thickness)), rot_axis, angle)
        hole = structures.Box(mat2, ((base_com[0]-a2/2., base_com[1]-b2/2., base_com[2]), (base_com[0]+a2/2., base_com[1]+b2/2., base_com[2]+thickness)), rot_axis, angle)
        return [sqr, hole]
  
class split_square_ring(object):
    def __new__(self, mat1, mat2, base_com, a1, a2, b1, b2, g1, thickness, gap_angle=0., angle=0., rot_axis=(0.,0.,1.)):
        r = np.sqrt(a1**2+b1**2)
        sqr  = structures.Box(mat1, ((base_com[0]-a1/2., base_com[1]-b1/2., base_com[2]), (base_com[0]+a1/2., base_com[1]+b1/2., base_com[2]+thickness)), rot_axis, angle)
        hole = structures.Box(mat2, ((base_com[0]-a2/2., base_com[1]-b2/2., base_com[2]), (base_com[0]+a2/2., base_com[1]+b2/2., base_com[2]+thickness)), rot_axis, angle)
        gap  = structures.Box(mat2, ((base_com[0], base_com[1]-g1/2., base_com[2]), (base_com[0]+r/2., base_com[1]+g1/2., base_com[2]+thickness)), rot_axis, angle+gap_angle, base_com)
        return [sqr, gap, hole]

class round_ring(object):
    def __new__(self, mat1, mat2, base_com, r1, r2, thickness, axis, angle=0., rot_axis=(0.,0.,1.)):
        cir  = structures.Cylinder(mat1, base_com, r1, thickness, axis, rot_axis, angle)
        hole = structures.Cylinder(mat2, base_com, r2, thickness, axis, rot_axis, angle)
        return [cir, hole]

class split_round_ring(object):
    def __new__(self, mat1, mat2, base_com, r1, r2, g1, thickness, axis, gap_angle=0., angle=0., rot_axis=(0.,0.,1.)):
        cir  = structures.Cylinder(mat1, base_com, r1, thickness, axis, rot_axis, angle)
        hole = structures.Cylinder(mat2, base_com, r2, thickness, axis, rot_axis, angle)
        gap  = structures.Box(mat2, ((base_com[0], base_com[1]-g1/2., base_com[2]), (base_com[0]+r1, base_com[1]+g1/2., base_com[2]+thickness)), rot_axis, angle+gap_angle, base_com)
        return [cir, gap, hole]

class cross(object):
    def __new__(self, mat1, base_com, a1, b1, thickness, angle=0., rot_axis=(0.,0.,1.)):     
        sqra  = structures.Box(mat1, ((base_com[0]-a1/2., base_com[1]-b1/2., base_com[2]), (base_com[0]+a1/2., base_com[1]+b1/2., base_com[2]+thickness)), rot_axis, angle)
        sqrb  = structures.Box(mat1, ((base_com[0]-b1/2., base_com[1]-a1/2., base_com[2]), (base_com[0]+b1/2., base_com[1]+a1/2., base_com[2]+thickness)), rot_axis, angle)       
        return [sqra, sqrb]

class cross2(object):
    def __new__(self, mat1, base_com, a1, b1, sqa, sqb, thickness, angle=0., rot_axis=(0.,0.,1.)):     
        sqrx  = structures.Box(mat1, ((base_com[0]-a1/2., base_com[1]-b1/2., base_com[2]), (base_com[0]+a1/2., base_com[1]+b1/2., base_com[2]+thickness)), rot_axis, angle)
        sqry  = structures.Box(mat1, ((base_com[0]-b1/2., base_com[1]-a1/2., base_com[2]), (base_com[0]+b1/2., base_com[1]+a1/2., base_com[2]+thickness)), rot_axis, angle)       
        sqr1  = structures.Box(mat1, ((base_com[0]-(b1+sqb)/2., base_com[1]-sqa/2., base_com[2]), (base_com[0]-(b1-sqb)/2., base_com[1]+sqa/2., base_com[2]+thickness)), rot_axis, angle)       
        sqr2  = structures.Box(mat1, ((base_com[0]-sqa/2., base_com[1]-(b1+sqb)/2., base_com[2]), (base_com[0]+sqa/2., base_com[1]-(b1-sqb)/2., base_com[2]+thickness)), rot_axis, angle)       
        sqr3  = structures.Box(mat1, ((base_com[0]+(b1-sqb)/2., base_com[1]-sqa/2., base_com[2]), (base_com[0]+(b1+sqb)/2., base_com[1]+sqa/2., base_com[2]+thickness)), rot_axis, angle)       
        sqr4  = structures.Box(mat1, ((base_com[0]-sqa/2., base_com[1]+(b1-sqb)/2., base_com[2]), (base_com[0]+sqa/2., base_com[1]+(b1+sqb)/2., base_com[2]+thickness)), rot_axis, angle)       


        return [sqrx, sqry, sqr1, sqr2, sqr3, sqr4]

class I_shape(object):
    def __new__(self, mat1, base_com, a1, a2, b1, b2, thickness, angle=0., rot_axis=(0.,0.,1.)):     
        sqry   = structures.Box(mat1, ((base_com[0]-a2/2., base_com[1]-b1/2., base_com[2]), (base_com[0]+a2/2., base_com[1]+b1/2., base_com[2]+thickness)), rot_axis, angle)
        sqrx1  = structures.Box(mat1, ((base_com[0]-a1/2., base_com[1]-b1/2., base_com[2]), (base_com[0]+a1/2., base_com[1]-b2/2., base_com[2]+thickness)), rot_axis, angle, base_com)
        sqrx2  = structures.Box(mat1, ((base_com[0]-a1/2., base_com[1]+b2/2., base_com[2]), (base_com[0]+a1/2., base_com[1]+b1/2., base_com[2]+thickness)), rot_axis, angle, base_com)        
        return [sqry, sqrx1, sqrx2]

class m_shape(object):
    def __new__(self, mat1, base_com, a1, a2, b1, b2, thickness, angle=0., rot_axis=(0.,0.,1.)):     
        sqry1  = structures.Box(mat1, ((base_com[0]-a2/2., base_com[1]-b1/2., base_com[2]), (base_com[0]+a2/2., base_com[1]+b1/2., base_com[2]+thickness)), rot_axis, angle)
        sqry2  = structures.Box(mat1, ((base_com[0]-a1/2., base_com[1]-b1/2., base_com[2]), (base_com[0]-a1/2.+a2, base_com[1]+b1/2., base_com[2]+thickness)), rot_axis, angle, base_com)
        sqry3  = structures.Box(mat1, ((base_com[0]+a1/2.-a2, base_com[1]-b1/2., base_com[2]), (base_com[0]+a1/2., base_com[1]+b1/2., base_com[2]+thickness)), rot_axis, angle, base_com)        
        sqrx   = structures.Box(mat1, ((base_com[0]-a1/2., base_com[1]-b1/2., base_com[2]), (base_com[0]+a1/2., base_com[1]-b1/2.+b2, base_com[2]+thickness)), rot_axis, angle, base_com)      
        return [sqry1, sqry2, sqry3, sqrx]

class S_shape(object):
    def __new__(self, mat1, mat2, base_com, a1, a2, b1, b2, thickness, angle=0., rot_axis=(0.,0.,1.)):
        bb = 3*b1+2*b2
        sqr  = structures.Box(mat1, ((base_com[0]-a1/2., base_com[1]-bb/2., base_com[2]), (base_com[0]+a1/2., base_com[1]+bb/2., base_com[2]+thickness)), rot_axis, angle)
        hole1 = structures.Box(mat2, ((base_com[0]-a1/2.+a2, base_com[1]-bb/2.+b1, base_com[2]), (base_com[0]+a1/2., base_com[1]-bb/2.+b1+b2, base_com[2]+thickness)), rot_axis, angle)
        hole2 = structures.Box(mat2, ((base_com[0]-a1/2., base_com[1]+b1/2., base_com[2]), (base_com[0]+a1/2.-a2, base_com[1]+b1/2.+b2, base_com[2]+thickness)), rot_axis, angle)
        return [sqr, hole1, hole2]

class square_cavity_1(object):
    def __new__(self, mat1, mat2, base_com, a1, a2, b1, b2, c1, c2, c3, d1, thickness, angle=0., rot_axis=(0.,0.,1.)):
        sqr  = structures.Box(mat1, ((base_com[0]-a1/2., base_com[1]-b1/2., base_com[2]), (base_com[0]+a1/2., base_com[1]+b1/2., base_com[2]+thickness)), rot_axis, angle)
        hole = structures.Box(mat2, ((base_com[0]-a2/2., base_com[1]-b2/2., base_com[2]), (base_com[0]+a2/2., base_com[1]+b2/2., base_com[2]+thickness)), rot_axis, angle)
        cen  = structures.Box(mat1, ((base_com[0]-d1/2., base_com[1]-b2/2., base_com[2]), (base_com[0]+d1/2., base_com[1]+b2/2., base_com[2]+thickness)), rot_axis, angle)
        cav  = structures.Box(mat1, ((base_com[0]-c1/2., base_com[1]-(2*c2+c3)/2., base_com[2]), (base_com[0]+c1/2., base_com[1]+(2*c2+c3)/2., base_com[2]+thickness)), rot_axis, angle)
        cavhole =structures.Box(mat2, ((base_com[0]-c1/2., base_com[1]-c3/2., base_com[2]), (base_com[0]+c1/2., base_com[1]+c3/2., base_com[2]+thickness)), rot_axis, angle)
        return [sqr, hole, cen, cav, cavhole]
  
class square_cavity_2(object):
    def __new__(self, mat1, mat2, base_com, a1, a2, b1, b2, c1, c2, c3, d1, thickness, angle=0., rot_axis=(0.,0.,1.)):
        sqr  = structures.Box(mat1, ((base_com[0]-a1/2., base_com[1]-b1/2., base_com[2]), (base_com[0]+a1/2., base_com[1]+b1/2., base_com[2]+thickness)), rot_axis, angle)
        hole = structures.Box(mat2, ((base_com[0]-a2/2., base_com[1]-b2/2., base_com[2]), (base_com[0]+a2/2., base_com[1]+b2/2., base_com[2]+thickness)), rot_axis, angle)
        cen  = structures.Box(mat1, ((base_com[0]-d1/2., base_com[1]-b2/2., base_com[2]), (base_com[0]+d1/2., base_com[1]+b2/2., base_com[2]+thickness)), rot_axis, angle)
        cav1  = structures.Box(mat1, ((base_com[0]-a1/2., base_com[1]-(2*c2+c3)/2., base_com[2]), (base_com[0]-a1/2.+c1, base_com[1]+(2*c2+c3)/2., base_com[2]+thickness)), rot_axis, angle)
        cavhole1 =structures.Box(mat2, ((base_com[0]-a1/2., base_com[1]-c3/2., base_com[2]), (base_com[0]-a1/2.+c1, base_com[1]+c3/2., base_com[2]+thickness)), rot_axis, angle)
        cav2  = structures.Box(mat1, ((base_com[0]+a1/2.-c1, base_com[1]-(2*c2+c3)/2., base_com[2]), (base_com[0]+a1/2., base_com[1]+(2*c2+c3)/2., base_com[2]+thickness)), rot_axis, angle)
        cavhole2 =structures.Box(mat2, ((base_com[0]+a1/2.-c1, base_com[1]-c3/2., base_com[2]), (base_com[0]+a1/2., base_com[1]+c3/2., base_com[2]+thickness)), rot_axis, angle)
        return [sqr, hole, cen, cav1, cavhole1, cav2, cavhole2]

class square_cavity_3(object):
    def __new__(self, mat1, mat2, base_com, a1, a2, b1, b2, c1, c2, c3, thickness, angle=0., rot_axis=(0.,0.,1.)):
        sqr  = structures.Box(mat1, ((base_com[0]-a1/2., base_com[1]-b1/2., base_com[2]), (base_com[0]+a1/2., base_com[1]+b1/2., base_com[2]+thickness)), rot_axis, angle)
        hole = structures.Box(mat2, ((base_com[0]-a2/2., base_com[1]-b2/2., base_com[2]), (base_com[0]+a2/2., base_com[1]+b2/2., base_com[2]+thickness)), rot_axis, angle)
        cav1  = structures.Box(mat1, ((base_com[0]+a1/2.-c1, base_com[1]-(2*c2+c3)/2., base_com[2]), (base_com[0]+a1/2., base_com[1]+(2*c2+c3)/2., base_com[2]+thickness)), rot_axis, angle)
        cavhole1 =structures.Box(mat2, ((base_com[0]+a1/2.-c1, base_com[1]-c3/2., base_com[2]), (base_com[0]+a1/2., base_com[1]+c3/2., base_com[2]+thickness)), rot_axis, angle)
        return [sqr, hole, cav1, cavhole1]

class square_cavity_4(object):
    def __new__(self, mat1, mat2, base_com, a1, a2, b1, b2, c1, c2, c3, thickness, angle=0., rot_axis=(0.,0.,1.)):
        sqr  = structures.Box(mat1, ((base_com[0]-a1/2., base_com[1]-b1/2., base_com[2]), (base_com[0]+a1/2., base_com[1]+b1/2., base_com[2]+thickness)), rot_axis, angle)
        hole = structures.Box(mat2, ((base_com[0]-a2/2., base_com[1]-b2/2., base_com[2]), (base_com[0]+a2/2., base_com[1]+b2/2., base_com[2]+thickness)), rot_axis, angle)
        cav1  = structures.Box(mat1, ((base_com[0]+a1/2.-c1, base_com[1]-(2*c2+c3)/2., base_com[2]), (base_com[0]+a1/2., base_com[1]+(2*c2+c3)/2., base_com[2]+thickness)), rot_axis, angle)
        cavhole1 =structures.Box(mat2, ((base_com[0]+a1/2.-c1, base_com[1]-c3/2., base_com[2]), (base_com[0]+a1/2., base_com[1]+c3/2., base_com[2]+thickness)), rot_axis, angle)
        return [sqr, hole, cav1, cavhole1]
    
class square_1(object):
    def __new__(self, mat1, mat2, base_com, a1, a2, b1, b2, g1, d1, thickness, angle=0., rot_axis=(0.,0.,1.)):
        sqr  = structures.Box(mat1, ((base_com[0]-a1/2., base_com[1]-b1/2., base_com[2]), (base_com[0]+a1/2., base_com[1]+b1/2., base_com[2]+thickness)), rot_axis, angle)
        hole = structures.Box(mat2, ((base_com[0]-a2/2., base_com[1]-b2/2., base_com[2]), (base_com[0]+a2/2., base_com[1]+b2/2., base_com[2]+thickness)), rot_axis, angle)
        sqrx   = structures.Box(mat1, ((base_com[0]-a1/2., base_com[1]-d1/2., base_com[2]), (base_com[0]+a1/2., base_com[1]+d1/2., base_com[2]+thickness)), rot_axis, angle)      
        sqry   = structures.Box(mat1, ((base_com[0]-d1/2., base_com[1]-b1/2., base_com[2]), (base_com[0]+d1/2., base_com[1]+b1/2., base_com[2]+thickness)), rot_axis, angle)
        sqrh1  = structures.Box(mat2, ((base_com[0]-np.sqrt(a1**2+b1**2)/2., base_com[1]-g1/2., base_com[2]), (base_com[0]+np.sqrt(a1**2+b1**2)/2., base_com[1]+g1/2., base_com[2]+thickness)), rot_axis, angle+np.pi/4., base_com)      
        sqrh2  = structures.Box(mat2, ((base_com[0]-g1/2., base_com[1]-np.sqrt(a1**2+b1**2)/2., base_com[2]), (base_com[0]+g1/2., base_com[1]+np.sqrt(a1**2+b1**2)/2., base_com[2]+thickness)), rot_axis, angle+np.pi/4., base_com)
        return [sqr, hole, sqrh1,sqrh2,sqrx,sqry]

class square_2(object):
    def __new__(self, mat1, mat2, base_com, a1, a2, b1, b2, g1, d1, thickness, angle=0., rot_axis=(0.,0.,1.)):
        sqr  = structures.Box(mat1, ((base_com[0]-a1/2., base_com[1]-b1/2., base_com[2]), (base_com[0]+a1/2., base_com[1]+b1/2., base_com[2]+thickness)), rot_axis, angle)
        hole = structures.Box(mat2, ((base_com[0]-a2/2., base_com[1]-b2/2., base_com[2]), (base_com[0]+a2/2., base_com[1]+b2/2., base_com[2]+thickness)), rot_axis, angle)
        sqrh1   = structures.Box(mat2, ((base_com[0]-a1/2., base_com[1]-d1/2., base_com[2]), (base_com[0]+a1/2., base_com[1]+d1/2., base_com[2]+thickness)), rot_axis, angle)      
        sqrh2   = structures.Box(mat2, ((base_com[0]-d1/2., base_com[1]-b1/2., base_com[2]), (base_com[0]+d1/2., base_com[1]+b1/2., base_com[2]+thickness)), rot_axis, angle)
        sqrx  = structures.Box(mat1, ((base_com[0]-np.sqrt(a1**2+b1**2)/2., base_com[1]-g1/2., base_com[2]), (base_com[0]+np.sqrt(a1**2+b1**2)/2., base_com[1]+g1/2., base_com[2]+thickness)), rot_axis, angle+np.pi/4., base_com)      
        sqry  = structures.Box(mat1, ((base_com[0]-g1/2., base_com[1]-np.sqrt(a1**2+b1**2)/2., base_com[2]), (base_com[0]+g1/2., base_com[1]+np.sqrt(a1**2+b1**2)/2., base_com[2]+thickness)), rot_axis, angle+np.pi/4., base_com)
        return [sqr, hole, sqrh1,sqrh2,sqrx,sqry]

class square_3(object):
    def __new__(self, mat1, mat2, base_com, a1, a2, b1, b2, a3, a4, b3, b4, g1, d1, thickness, angle=0., rot_axis=(0.,0.,1.)):
        sqr1  = structures.Box(mat1, ((base_com[0]-a1/2., base_com[1]-b1/2., base_com[2]), (base_com[0]+a1/2., base_com[1]+b1/2., base_com[2]+thickness)), rot_axis, angle)
        hole1 = structures.Box(mat2, ((base_com[0]-a2/2., base_com[1]-b2/2., base_com[2]), (base_com[0]+a2/2., base_com[1]+b2/2., base_com[2]+thickness)), rot_axis, angle)
        sqr2  = structures.Box(mat1, ((base_com[0]-a3/2., base_com[1]-b3/2., base_com[2]), (base_com[0]+a3/2., base_com[1]+b3/2., base_com[2]+thickness)), rot_axis, angle)
        hole2 = structures.Box(mat2, ((base_com[0]-a4/2., base_com[1]-b4/2., base_com[2]), (base_com[0]+a4/2., base_com[1]+b4/2., base_com[2]+thickness)), rot_axis, angle)
        sqrx   = structures.Box(mat1, ((base_com[0]-a1/2., base_com[1]-d1/2., base_com[2]), (base_com[0]+a1/2., base_com[1]+d1/2., base_com[2]+thickness)), rot_axis, angle)      
        sqry   = structures.Box(mat1, ((base_com[0]-d1/2., base_com[1]-b1/2., base_com[2]), (base_com[0]+d1/2., base_com[1]+b1/2., base_com[2]+thickness)), rot_axis, angle)
        sqrh1  = structures.Box(mat2, ((base_com[0]-np.sqrt(a3**2+b3**2)/2., base_com[1]-g1/2., base_com[2]), (base_com[0]+np.sqrt(a3**2+b3**2)/2., base_com[1]+g1/2., base_com[2]+thickness)), rot_axis, angle+np.pi/4., base_com)      
        sqrh2  = structures.Box(mat2, ((base_com[0]-g1/2., base_com[1]-np.sqrt(a3**2+b3**2)/2., base_com[2]), (base_com[0]+g1/2., base_com[1]+np.sqrt(a3**2+b3**2)/2., base_com[2]+thickness)), rot_axis, angle+np.pi/4., base_com)
        return [sqr1, hole1, sqrx, sqry, sqr2, hole2, sqrh1,sqrh2,]
    
class round_1(object):
    def __new__(self, mat1, mat2, base_com, r1, r2, g1, a1, thickness, axis, angle=0., rot_axis=(0.,0.,1.)):

        cir  = structures.Cylinder(mat1, base_com, r1, thickness, axis, rot_axis, angle)
        sqrx   = structures.Box(mat1, ((base_com[0]-r1,    base_com[1]-a1/2., base_com[2]), (base_com[0]+r1, base_com[1]+a1/2., base_com[2]+thickness)), rot_axis, angle)      
        sqry   = structures.Box(mat1, ((base_com[0]-a1/2., base_com[1]-r1, base_com[2]), (base_com[0]+a1/2., base_com[1]+r1, base_com[2]+thickness)), rot_axis, angle)
        sqrh1  = structures.Box(mat2, ((base_com[0]-r1,    base_com[1]-a1/2., base_com[2]), (base_com[0]+r1, base_com[1]+a1/2., base_com[2]+thickness)), rot_axis, angle+np.pi/4., base_com)      
        sqrh2  = structures.Box(mat2, ((base_com[0]-a1/2., base_com[1]-r1, base_com[2]), (base_com[0]+a1/2., base_com[1]+r1, base_com[2]+thickness)), rot_axis, angle+np.pi/4., base_com)

        hole = structures.Cylinder(mat2, base_com, r2, thickness, axis, rot_axis, angle)

        return [cir,hole,sqrh1,sqrh2,sqrx,sqry]

class fishnet(object):
    def __new__(self, mat1, mat2, base_com, a1, a2, a3, b1, b2, b3, thickness, angle=0., rot_axis=(0.,0.,1.)):

        sqrx1   = structures.Box(mat1, ((base_com[0]-a3-a2, base_com[1]-b1/2., base_com[2]), (base_com[0]-a3,    base_com[1]+b1/2., base_com[2]+thickness)), rot_axis, angle)      
        sqrx2   = structures.Box(mat1, ((base_com[0]+a3,    base_com[1]-b1/2., base_com[2]), (base_com[0]+a3+a2, base_com[1]+b1/2., base_com[2]+thickness)), rot_axis, angle)      
        sqry1   = structures.Box(mat1, ((base_com[0]-a1/2., base_com[1]-b3-b2, base_com[2]), (base_com[0]+a1/2., base_com[1]-b3,    base_com[2]+thickness)), rot_axis, angle)      
        sqry2   = structures.Box(mat1, ((base_com[0]-a1/2., base_com[1]+b3,    base_com[2]), (base_com[0]+a1/2., base_com[1]+b3+b2, base_com[2]+thickness)), rot_axis, angle)      

        return [sqrx1,sqrx2,sqry1,sqry2]

class dart(object):
    def __new__(self, mat1, base_com, a1, b1, d1, thickness, angle=0., rot_axis=(0.,0.,1.)):
        sqrx1   = structures.Box(mat1, ((base_com[0]-b1+d1/2., base_com[1]+a1/2.-d1, base_com[2]), (base_com[0]+d1/2.,    base_com[1]+a1/2.,    base_com[2]+thickness)), rot_axis, angle)      
        sqrx2   = structures.Box(mat1, ((base_com[0]-a1/2.,    base_com[1]-d1/2.,    base_com[2]), (base_com[0]+a1/2.,    base_com[1]+d1/2.,    base_com[2]+thickness)), rot_axis, angle)      
        sqrx3   = structures.Box(mat1, ((base_com[0]-d1/2.,    base_com[1]-a1/2.,    base_com[2]), (base_com[0]+a1-d1/2., base_com[1]-a1/2.+d1, base_com[2]+thickness)), rot_axis, angle)      

        return [sqrx1,sqrx2,sqrx3]




'''    
def split_ring_resonator2(a1,a2,b1,b2,b3,b4,mat1,mat2,thickness,lx,ly,lz,dx,dy,dz):
       
    sqr  = structures.Box(mat1, (((lx-a1)/2-dx/2,(ly-b1-b2-b3)/2-dy/2,(lz-thickness)/2-dz/2),((lx+a1)/2-dx/2,(ly+b1+b2+b3)/2-dy/2,(lz+thickness)/2-dz/2)))
    hole1 = structures.Box(mat2, (((lx-a2)/2-dx/2,(ly-b4)/2-dy/2,(lz-thickness)/2-dz/2),((lx+a2)/2-dx/2,(ly+b4)/2-dy/2,(lz+thickness)/2-dz/2)))
    hole2 = structures.Box(mat2, (((lx-dx)/2,(ly+b1-b2-b3-dy)/2,(lz-thickness)/2-dz/2),((lx+a1)/2-dx/2,(ly+b1+b2-b3-dy)/2,(lz+thickness)/2-dz/2)))
    
    return [sqr,hole1,hole2]


def sqrpatch(a1,b1,mat1,thickness,lx,ly,lz,dx,dy,dz):
    sqr = structures.Box(mat1, (((lx-a1)/2-dx/2,(ly-b1)/2-dy/2,(lz-thickness)/2-dz/2),((lx+a1)/2-dx/2,(ly+b1)/2-dy/2,(lz+thickness)/2-dz/2)))

    return [sqr]
'''
