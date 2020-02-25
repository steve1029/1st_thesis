from KEMP import structures
#from KEMP import structures_new as structures
import numpy as np

class square_ring(object):
	def __new__(self, mat1, mat2, base_com, a1, a2, b1, b2, thick, angle=0., rot_axis=(0.,0.,1.)):

		xx = base_com[0]
		yy = base_com[1]
		zz = base_com[2]

		sqr  = structures.Box(mat1, ((xx-a1/2., yy-b1/2., zz), (xx+a1/2., yy+b1/2., zz+thick)), rot_axis, angle)
		hole = structures.Box(mat2, ((xx-a2/2., yy-b2/2., zz), (xx+a2/2., yy+b2/2., zz+thick)), rot_axis, angle)

		#sqr  = structures.Box(mat1, ((xx-a1/2., yy-b1/2., zz), (xx+a1/2., yy+b1/2., zz+thick)))
		#hole = structures.Box(mat2, ((xx-a2/2., yy-b2/2., zz), (xx+a2/2., yy+b2/2., zz+thick)))

		return [sqr, hole]
  
class split_square_ring(object):
	def __new__(self, mat1, mat2, base_com, a1, a2, b1, b2, g1, thick, gap_angle=0., angle=0., rot_axis=(0.,0.,1.)):

		xx = base_com[0]
		yy = base_com[1]
		zz = base_com[2]

		r	 = np.sqrt(a1**2+b1**2)
		sqr  = structures.Box(mat1, ((xx-a1/2., yy-b1/2., zz), (xx+a1/2., yy+b1/2., zz+thick)), rot_axis, angle)
		hole = structures.Box(mat2, ((xx-a2/2., yy-b2/2., zz), (xx+a2/2., yy+b2/2., zz+thick)), rot_axis, angle)
		gap  = structures.Box(mat2, ((xx,		yy-g1/2., zz), (xx+r /2., yy+g1/2., zz+thick)), rot_axis, angle+gap_angle, base_com)

		#sqr  = structures.Box(mat1, ((xx-a1/2., yy-b1/2., zz), (xx+a1/2., yy+b1/2., zz+thick)))
		#hole = structures.Box(mat2, ((xx-a2/2., yy-b2/2., zz), (xx+a2/2., yy+b2/2., zz+thick)))
		#gap  = structures.Box(mat2, ((xx,		 yy-g1/2., zz), (xx+r /2., yy+g1/2., zz+thick)))

		return [sqr, gap, hole]

class round_ring(object):
	def __new__(self, mat1, mat2, base_com, r1, r2, thick, axis, angle=0., rot_axis=(0.,0.,1.)):

		cir  = structures.Cylinder(mat1, base_com, r1, thick, axis, rot_axis, angle)
		hole = structures.Cylinder(mat2, base_com, r2, thick, axis, rot_axis, angle)

		#cir  = structures.Cylinder(mat1, base_com, r1, thick, axis, rot_axis, angle)
		#hole = structures.Cylinder(mat2, base_com, r2, thick, axis, rot_axis, angle)

		return [cir, hole]

class split_round_ring(object):
	def __new__(self, mat1, mat2, base_com, r1, r2, g1, thick, axis, gap_angle=0., angle=0., rot_axis=(0.,0.,1.)):

		xx = base_com[0]
		yy = base_com[1]
		zz = base_com[2]

		cir  = structures.Cylinder(mat1, base_com, r1, thick, axis, rot_axis, angle)
		hole = structures.Cylinder(mat2, base_com, r2, thick, axis, rot_axis, angle)

		gap  = structures.Box(mat2, ((xx, yy-g1/2., zz), (xx+r1, yy+g1/2., zz+thick)), rot_axis, angle+gap_angle, base_com)

		return [cir, gap, hole]

class cross(object):
	def __new__(self, mat1, base_com, a1, b1, thick, angle=0., rot_axis=(0.,0.,1.)):	 
		"""
				  b
				 _____				_ _
				|	  |				 |
				|	  |				 |
		 _______|	  |______		 |
		|				  	 |		 | 
		|					 | b	 | a
		|_______	   ______|		 |
				|	  |				 |
				|	  |				 |
				|_____|				_|_

		<-------------------->
				   a
		
		"""

		xx = base_com[0]
		yy = base_com[1]
		zz = base_com[2]

		sqra  = structures.Box(mat1, ((xx-a1/2., yy-b1/2., zz), (xx+a1/2., yy+b1/2., zz+thick)), rot_axis, angle)
		sqrb  = structures.Box(mat1, ((xx-b1/2., yy-a1/2., zz), (xx+b1/2., yy+a1/2., zz+thick)), rot_axis, angle)		  
		return [sqra, sqrb]

class cross_4arms(object):
	def __new__(self, mat, base_com, a, b, sqa, sqb, thick, angle=0., rot_axis=(0.,0.,1.)):	   
		"""
					 <---- a2 --->
					 _____________
					|	  		  | b2
					|_____________|		    	  _ _
						|     |				       |
			 ___		|	  |		  ___	_ _    |
			|   |_______|	  |______|	 |	 |	   |
			|	|		base_com  	 |	 |	 |	   |	 
			|	|   	   .	  b1 | 	 |	 | a2  | a1
			|	|_______	   ______|	 |	 |	   |
			|___|		|	  |		 |___|  _|_	   |
						|     |		   b2	 	   |
					 ___|_____|___				  _|_
					|	<-b1 ->	  | b2
					|_____________|

					<---- a2 ----->

				<-------------------->
						   a1
		
		"""

		xx = base_com[0]
		yy = base_com[1]
		zz = base_com[2]

		structure = []

		vertical_bar   = structures.Box(mat, ((xx-a/2.,yy-b/2.,zz), (xx+a/2.,yy+b/2.,zz+thick)), rot_axis, angle)
		horizental_bar = structures.Box(mat, ((xx-b/2.,yy-a/2.,zz), (xx+b/2.,yy+a/2.,zz+thick)), rot_axis, angle)		  

		right_col = structures.Box(mat, ((xx-(b+sqb)/2., yy-sqa/2., zz), (xx-(b-sqb)/2., yy+sqa/2., zz+thick)), rot_axis, angle)	  
		left_col  = structures.Box(mat, ((xx+(b-sqb)/2., yy-sqa/2., zz), (xx+(b+sqb)/2., yy+sqa/2., zz+thick)), rot_axis, angle)	  
		under_row = structures.Box(mat, ((xx-sqa/2., yy-(b+sqb)/2., zz), (xx+sqa/2., yy-(b-sqb)/2., zz+thick)), rot_axis, angle)	  
		upper_row = structures.Box(mat, ((xx-sqa/2., yy+(b-sqb)/2., zz), (xx+sqa/2., yy+(b+sqb)/2., zz+thick)), rot_axis, angle) 

		structure.append(vertical_bar)
		structure.append(horizental_bar)
		structure.append(right_col)
		structure.append(left_col)
		structure.append(under_row)
		structure.append(upper_row)

		return structure

class I_shape(object):
	def __new__(self, mat1, base_com, a1, a2, b1, b2, thick, angle=0., rot_axis=(0.,0.,1.)):	 

		xx = base_com[0]
		yy = base_com[1]
		zz = base_com[2]

		sqry  = structures.Box(mat1, ((xx-a2/2., yy-b1/2., zz), (xx+a2/2., yy+b1/2., zz+thick)), rot_axis, angle)
		sqrx1 = structures.Box(mat1, ((xx-a1/2., yy-b1/2., zz), (xx+a1/2., yy-b2/2., zz+thick)), rot_axis, angle, base_com)
		sqrx2 = structures.Box(mat1, ((xx-a1/2., yy+b2/2., zz), (xx+a1/2., yy+b1/2., zz+thick)), rot_axis, angle, base_com)

		return [sqry, sqrx1, sqrx2]

class m_shape(object):
	def __new__(self, mat1, base_com, a1, a2, b1, b2, thick, angle=0., rot_axis=(0.,0.,1.)):	 

		xx = base_com[0]
		yy = base_com[1]
		zz = base_com[2]

		sqry1 = structures.Box(mat1, ((xx-a2/2., yy-b1/2., zz), (xx+a2/2., yy+b1/2., zz+thick)), rot_axis, angle)
		sqry2 = structures.Box(mat1, ((xx-a1/2., yy-b1/2., zz), (xx-a1/2.+a2, yy+b1/2., zz+thick)), rot_axis, angle, base_com)
		sqry3 = structures.Box(mat1, ((xx+a1/2.-a2, yy-b1/2., zz), (xx+a1/2., yy+b1/2., zz+thick)), rot_axis, angle, base_com) 
		sqrx  = structures.Box(mat1, ((xx-a1/2., yy-b1/2., zz), (xx+a1/2., yy-b1/2.+b2, zz+thick)), rot_axis, angle, base_com) 

		return [sqry1, sqry2, sqry3, sqrx]

class two_shape(object):
	def __new__(self, mat1, mat2, base_com, a1, a2, b1, b2, thick, angle=0., rot_axis=(0.,0.,1.)):

		xx = base_com[0]
		yy = base_com[1]
		zz = base_com[2]

		bb = 3*b1+2*b2
		sqr   = structures.Box(mat1, ((xx-a1/2., yy-bb/2., zz), (xx+a1/2., yy+bb/2., zz+thick)), rot_axis, angle)
		hole1 = structures.Box(mat2, ((xx-a1/2.+a2, yy-bb/2.+b1, zz), (xx+a1/2., yy-bb/2.+b1+b2, zz+thick)), rot_axis, angle)
		hole2 = structures.Box(mat2, ((xx-a1/2., yy+b1/2., zz), (xx+a1/2.-a2, yy+b1/2.+b2, zz+thick)), rot_axis, angle)

		return [sqr, hole1, hole2]

class square_twohammers(object):
	def __new__(self, mat1, mat2, base_com, a1, a2, b1, b2, c1, c2, c3, d1, thick, angle=0., rot_axis=(0.,0.,1.)):

		xx = base_com[0]
		yy = base_com[1]
		zz = base_com[2]

		sqr  = structures.Box(mat1, ((xx-a1/2., yy-b1/2., zz), (xx+a1/2., yy+b1/2., zz+thick)), rot_axis, angle)
		hole = structures.Box(mat2, ((xx-a2/2., yy-b2/2., zz), (xx+a2/2., yy+b2/2., zz+thick)), rot_axis, angle)
		cen  = structures.Box(mat1, ((xx-d1/2., yy-b2/2., zz), (xx+d1/2., yy+b2/2., zz+thick)), rot_axis, angle)
		cav  = structures.Box(mat1, ((xx-c1/2., yy-(2*c2+c3)/2., zz), (xx+c1/2., yy+(2*c2+c3)/2., zz+thick)), rot_axis, angle)
		cavhole =structures.Box(mat2, ((xx-c1/2., yy-c3/2., zz), (xx+c1/2., yy+c3/2., zz+thick)), rot_axis, angle)

		return [sqr, hole, cen, cav, cavhole]
  
class square_column_arms(object):
	def __new__(self, mat1, mat2, base_com, a1, a2, b1, b2, c1, c2, c3, d1, thick, angle=0., rot_axis=(0.,0.,1.)):

		xx = base_com[0]
		yy = base_com[1]
		zz = base_com[2]

		sqr  = structures.Box(mat1, ((xx-a1/2., yy-b1/2., zz), (xx+a1/2., yy+b1/2., zz+thick)), rot_axis, angle)
		hole = structures.Box(mat2, ((xx-a2/2., yy-b2/2., zz), (xx+a2/2., yy+b2/2., zz+thick)), rot_axis, angle)
		cen  = structures.Box(mat1, ((xx-d1/2., yy-b2/2., zz), (xx+d1/2., yy+b2/2., zz+thick)), rot_axis, angle)

		cav1     = structures.Box(mat1, ((xx-a1/2., yy-(2*c2+c3)/2., zz), (xx-a1/2.+c1, yy+(2*c2+c3)/2., zz+thick)), rot_axis, angle)
		cavhole1 = structures.Box(mat2, ((xx-a1/2.,	yy-c3/2.       , zz), (xx-a1/2.+c1, yy+c3/2.       , zz+thick)), rot_axis, angle)

		cav2     = structures.Box(mat1, ((xx+a1/2.-c1, yy-(2*c2+c3)/2., zz), (xx+a1/2., yy+(2*c2+c3)/2., zz+thick)), rot_axis, angle)
		cavhole2 = structures.Box(mat2, ((xx+a1/2.-c1, yy-c3/2.       , zz), (xx+a1/2., yy+c3/2.       , zz+thick)), rot_axis, angle)

		return [sqr, hole, cen, cav1, cavhole1, cav2, cavhole2]

class square_cavity_3(object):
	def __new__(self, mat1, mat2, base_com, a1, a2, b1, b2, c1, c2, c3, thick, angle=0., rot_axis=(0.,0.,1.)):

		xx = base_com[0]
		yy = base_com[1]
		zz = base_com[2]

		sqr   = structures.Box(mat1, ((xx-a1/2.,    yy-b1/2.,		 zz), (xx+a1/2., yy+b1/2., zz+thick)), rot_axis, angle)
		hole  = structures.Box(mat2, ((xx-a2/2.,    yy-b2/2.,		 zz), (xx+a2/2., yy+b2/2., zz+thick)), rot_axis, angle)
		cav1  = structures.Box(mat1, ((xx+a1/2.-c1, yy-(2*c2+c3)/2., zz), (xx+a1/2., yy+(2*c2+c3)/2., zz+thick)), rot_axis, angle)
		cavhole1 = structures.Box(mat2, ((xx+a1/2.-c1, yy-c3/2.,  zz), (xx+a1/2., yy+c3/2., zz+thick)), rot_axis, angle)

		return [sqr, hole, cav1, cavhole1]

class square_cavity_4(object):
	def __new__(self, mat1, mat2, base_com, a1, a2, b1, b2, c1, c2, c3, thick, angle=0., rot_axis=(0.,0.,1.)):

		xx = base_com[0]
		yy = base_com[1]
		zz = base_com[2]

		sqr  = structures.Box(mat1, ((xx-a1/2., yy-b1/2., zz), (xx+a1/2., yy+b1/2., zz+thick)), rot_axis, angle)
		hole = structures.Box(mat2, ((xx-a2/2., yy-b2/2., zz), (xx+a2/2., yy+b2/2., zz+thick)), rot_axis, angle)
		cav1 = structures.Box(mat1, ((xx+a1/2.-c1, yy-(2*c2+c3)/2., zz), (xx+a1/2., yy+(2*c2+c3)/2., zz+thick)), rot_axis, angle)
		cavhole1 = structures.Box(mat2, ((xx+a1/2.-c1, yy-c3/2., zz), (xx+a1/2., yy+c3/2., zz+thick)), rot_axis, angle)

		return [sqr, hole, cav1, cavhole1]
	
class hollowsquare_insidecross(object):
	def __new__(self, mat1, mat2, base_com, a1, a2, b1, b2, g1, d1, thick, angle=0., rot_axis=(0.,0.,1.)):

		xx = base_com[0]
		yy = base_com[1]
		zz = base_com[2]

		sqr   = structures.Box(mat1, ((xx-a1/2., yy-b1/2., zz), (xx+a1/2., yy+b1/2., zz+thick)), rot_axis, angle)
		hole  = structures.Box(mat2, ((xx-a2/2., yy-b2/2., zz), (xx+a2/2., yy+b2/2., zz+thick)), rot_axis, angle)
		sqrx  = structures.Box(mat1, ((xx-a1/2., yy-d1/2., zz), (xx+a1/2., yy+d1/2., zz+thick)), rot_axis, angle)	  
		sqry  = structures.Box(mat1, ((xx-d1/2., yy-b1/2., zz), (xx+d1/2., yy+b1/2., zz+thick)), rot_axis, angle)
		sqrh1 = structures.Box(mat2, ((xx-np.sqrt(a1**2+b1**2)/2., yy-g1/2., zz), (xx+np.sqrt(a1**2+b1**2)/2., yy+g1/2., zz+thick)), rot_axis, angle+np.pi/4., base_com)		 
		sqrh2 = structures.Box(mat2, ((xx-g1/2., yy-np.sqrt(a1**2+b1**2)/2., zz), (xx+g1/2., yy+np.sqrt(a1**2+b1**2)/2., zz+thick)), rot_axis, angle+np.pi/4., base_com)

		return [sqr, hole, sqrh1,sqrh2,sqrx,sqry]

class hollowsquare_insidex(object):
	def __new__(self, mat1, mat2, base_com, a1, a2, b1, b2, g1, d1, thick, angle=0., rot_axis=(0.,0.,1.)):

		xx = base_com[0]
		yy = base_com[1]
		zz = base_com[2]

		sqr   = structures.Box(mat1, ((xx-a1/2., yy-b1/2., zz), (xx+a1/2., yy+b1/2., zz+thick)), rot_axis, angle)
		hole  = structures.Box(mat2, ((xx-a2/2., yy-b2/2., zz), (xx+a2/2., yy+b2/2., zz+thick)), rot_axis, angle)
		sqrh1 = structures.Box(mat2, ((xx-a1/2., yy-d1/2., zz), (xx+a1/2., yy+d1/2., zz+thick)), rot_axis, angle)	   
		sqrh2 = structures.Box(mat2, ((xx-d1/2., yy-b1/2., zz), (xx+d1/2., yy+b1/2., zz+thick)), rot_axis, angle)
		sqrx  = structures.Box(mat1, ((xx-np.sqrt(a1**2+b1**2)/2., yy-g1/2., zz), (xx+np.sqrt(a1**2+b1**2)/2., yy+g1/2., zz+thick)), rot_axis, angle+np.pi/4.)		
		sqry  = structures.Box(mat1, ((xx-g1/2., yy-np.sqrt(a1**2+b1**2)/2., zz), (xx+g1/2., yy+np.sqrt(a1**2+b1**2)/2., zz+thick)), rot_axis, angle+np.pi/4.)

		return [sqr, hole, sqrh1,sqrh2,sqrx,sqry]

class hollowsquare_4innerhammers(object):
	def __new__(self, mat1, mat2, base_com, a1, a2, b1, b2, a3, a4, b3, b4, g1, d1, thick, angle=0., rot_axis=(0.,0.,1.)):

		xx = base_com[0]
		yy = base_com[1]
		zz = base_com[2]

		sqr1   = structures.Box(mat1, ((xx-a1/2., yy-b1/2., zz), (xx+a1/2., yy+b1/2., zz+thick)), rot_axis, angle)
		hole1  = structures.Box(mat2, ((xx-a2/2., yy-b2/2., zz), (xx+a2/2., yy+b2/2., zz+thick)), rot_axis, angle)

		sqrx   = structures.Box(mat1, ((xx-a1/2., yy-d1/2., zz), (xx+a1/2., yy+d1/2., zz+thick)), rot_axis, angle)	  
		sqry   = structures.Box(mat1, ((xx-d1/2., yy-b1/2., zz), (xx+d1/2., yy+b1/2., zz+thick)), rot_axis, angle)

		sqr2   = structures.Box(mat1, ((xx-a3/2., yy-b3/2., zz), (xx+a3/2., yy+b3/2., zz+thick)), rot_axis, angle)
		hole2  = structures.Box(mat2, ((xx-a4/2., yy-b4/2., zz), (xx+a4/2., yy+b4/2., zz+thick)), rot_axis, angle)

		sqrh1  = structures.Box(mat2, ((xx-np.sqrt(a3**2+b3**2)/2., yy-g1/2., zz), (xx+np.sqrt(a3**2+b3**2)/2., yy+g1/2., zz+thick)), rot_axis, angle+np.pi/4.)		 
		sqrh2  = structures.Box(mat2, ((xx-g1/2., yy-np.sqrt(a3**2+b3**2)/2., zz), (xx+g1/2., yy+np.sqrt(a3**2+b3**2)/2., zz+thick)), rot_axis, angle+np.pi/4.)

		return [sqr1, hole1, sqrx, sqry, sqr2, hole2, sqrh1,sqrh2,]
	
class hollowcylinder_insidecross(object):
	def __new__(self, mat1, mat2, base_com, r1, r2, g1, a1, thick, axis, angle=0., rot_axis=(0.,0.,1.)):
		"""
				  __
			     | |
               _| |
			 _|  |
			|  |
		   |  |
			|_ |_			
   			  |_ |
 			    | | 
			     |_|

		"""

		xx = base_com[0]
		yy = base_com[1]
		zz = base_com[2]

		cir       = structures.Cylinder(mat1, base_com, r1, thick, axis, rot_axis, angle)
		cir_hole  = structures.Cylinder(mat2, base_com, r2, thick, axis, rot_axis, angle)
		spliter1  = structures.Box(mat2, ((xx-r1,    yy-a1/2., zz), (xx+r1, yy+a1/2., zz+thick)), rot_axis, angle+np.pi/4., base_com)	  
		spliter2  = structures.Box(mat2, ((xx-a1/2., yy-r1, zz), (xx+a1/2., yy+r1, zz+thick)), rot_axis, angle+np.pi/4., base_com)
		hori_box  = structures.Box(mat1, ((xx-r1,    yy-a1/2., zz), (xx+r1, yy+a1/2., zz+thick)), rot_axis, angle)	   
		verti_box = structures.Box(mat1, ((xx-a1/2., yy-r1, zz), (xx+a1/2., yy+r1, zz+thick)), rot_axis, angle)


		return [cir, cir_hole, splitter1, splitter2, hori_box, verti_box]

class fishnet(object):
	def __new__(self, mat1, mat2, center, a1, a2, a3, b1, b2, b3, thick, angle=0., rot_axis=(0.,0.,1.)):
		"""Fishnet structure.
		'base_com' is a center point of the bottom of the box.
		Bottom is lying in the xy-plane. In KEMP, we assume that plane wave is
		propagating along z-axis which means poynting vector is z-direction.
		The direction of E and H field of the plane wave is +Ey and -Hx.
		
				y
				^
				|
				|
		x <-----|
		 _____________________________________________________________		   _ _
		|            |        |               |        |              |			|
		|            |        |               |        |              |			|
		|            |        |               |        |              |			|
		|            |        |               |        |              |			|
		|____________|        |_______________|        |______________|	     	|
		|                                                             |	   	    |
		|                                                             |	b2 	    |
		|____________          _______________          ______________|   _ _   |  
		|            |        |               |	   	   | 			  |	   |    | 
		|			 |		  |	center(x,y,z) |		   |			  |	   |	|
		|			 |		  |		  .		  |		   |			  |	   | b3	| b1
		|			 |		  |		  		  |  	   |			  |	   |	|
		|____________|		  |_______________|		   |______________|	  _|_	|
		|                                                             |	    	|
		|                                                             | b2    	|
		|____________          _______________          ______________|	     	|
		|            |        |               |        |              |			|
		|            |        |               |        |              |			|
		|            |        |               |        |              |			|
		|            |        |               |        |              |			|
		|____________|________|_______________|________|______________|		   _|_

					  <- a2 -> <---- a3  ----> <- a2 ->

		<----------------------------- a1 ---------------------------->

		"""

		xx = center[0]
		yy = center[1]
		zz = center[2]
	
		structure_list = []

		background = structures.Box(mat1, ((xx-a1/2   , yy-b1/2   , zz), (xx+a1/2   , yy+b1/2   , zz+thick)), rot_axis, angle)

		left_col   = structures.Box(mat2, ((xx+a3/2   , yy-b1/2   , zz), (xx+a3/2+a2, yy+b1/2   , zz+thick)), rot_axis, angle)
		right_col  = structures.Box(mat2, ((xx-a3/2-a2, yy-b1/2   , zz), (xx-a3/2   , yy+b1/2   , zz+thick)), rot_axis, angle)
		upper_row  = structures.Box(mat2, ((xx-a1/2   , yy+b3/2   , zz), (xx+a1/2   , yy+b3/2+b2, zz+thick)), rot_axis, angle)
		under_row  = structures.Box(mat2, ((xx-a1/2   , yy-b3/2-b2, zz), (xx+a1/2   , yy-b3/2   , zz+thick)), rot_axis, angle)

		structure_list.append(background)
		structure_list.append(left_col)
		structure_list.append(right_col)
		structure_list.append(upper_row)
		structure_list.append(under_row)

		return structure_list

class dart(object):
	def __new__(self, mat1, base_com, a1, b1, d1, thick, angle=0., rot_axis=(0.,0.,1.)):

		xx = base_com[0]
		yy = base_com[1]
		zz = base_com[2]

		#sqrx1 = structures.Box(mat1, ((xx-b1+d1/2., yy+a1/2.-d1, zz), (xx+d1/2.   , yy+a1/2.   , zz+thick))) 
		#sqrx2 = structures.Box(mat1, ((xx-a1/2.   , yy-d1/2.   , zz), (xx+a1/2.   , yy+d1/2.   , zz+thick))) 
		#sqrx3 = structures.Box(mat1, ((xx-d1/2.   , yy-a1/2.   , zz), (xx+a1-d1/2., yy-a1/2.+d1, zz+thick))) 

		sqrx1 = structures.Box(mat1, ((xx-b1+d1/2., yy+a1/2.-d1, zz), (xx+d1/2.   , yy+a1/2.   , zz+thick)), rot_axis, angle) 
		sqrx2 = structures.Box(mat1, ((xx-a1/2.,    yy-d1/2.   , zz), (xx+a1/2.   , yy+d1/2.   , zz+thick)), rot_axis, angle) 
		sqrx3 = structures.Box(mat1, ((xx-d1/2.,    yy-a1/2.   , zz), (xx+a1-d1/2., yy-a1/2.+d1, zz+thick)), rot_axis, angle) 

		return [sqrx1,sqrx2,sqrx3]

class vertical_rectangles(object):
	def __new__(self, mat, N, a, b, thick,y_spacing, xycenter_of_bottom_3Dbox, rot_axis, angle):
		"""Vertical align of rectangle 3D boxes.

				 _______
				|		|
				|	.	|
				|_______|	
				 _______
				|		|
				|	.	|
			|	|_______|		y
	spacing	|	 _______		^
			|	|		|		|
			|	|	.	|	b	|
				|_______|		|
					a			|
								|
				x <-------------

		PARAMETERS
		----------
		mat	:	material object
			material object which is declared in KEMP.material
		N	: int
			The number of 3D boxes.
		a	: float
			x length of each 3D box.
		b	: float
			y length of each 3D box.
		thick	: float
			z lenfth of each 3D box.

		y_spacing : float
			spacing between the center of each 3D box along y-axis.

		xycenter_of_bottom_3Dbox : list or tuple with 3 float elements
			coordinates of the center of bottom xy plane of a 3D box.

		rot_axis	: list or tuple with length 3.
			choose the rotation angle.

		angle	: float
			rotation angle as radian.

		RETURN
		------
		structure_list : list
			a list which has structure objects as its elements.
		"""

		xx = xycenter_of_bottom_3Dbox[0]
		yy = xycenter_of_bottom_3Dbox[1]
		zz = xycenter_of_bottom_3Dbox[2]

		structure_list = []

		for n in range(N):
			structure_list.append( structures.Box(mat, (xx-a/2, yy+n*y_spacing-b/2, zz),(xx+a/2, yy+n*y_spacing+b/2, zz+thick)))

		return structure_list

		
class horizental_rectangles(object):
	def __new__(self, mat, N, a, b, thick, x_spacing, xycenter_of_bottom_3Dbox, rot_axis, angle):
		"""Horizental align of rectangle 3D boxes.

										a			y
			 _______	 _______	 _______		^
			|		|	|		|	|		|		|
			|	.	|	|	.	|	|	.	|	b	|
			|_______|	|_______|	|_______|		|
													|
							<----------->			|
								spacing				|
													|
		x	<---------------------------------------


		PARAMETERS
		----------
		mat	:	material object
			material object which is declared in KEMP.material
		N	: int
			The number of 3D boxes.
		a	: float
			x length of each 3D box.
		b	: float
			y length of each 3D box.
		thick	: float
			z lenfth of each 3D box.

		x_spacing : float
			spacing between the center of each 3D box along x-axis.

		xycenter_of_bottom_3Dbox : list or tuple with 3 float elements
			coordinates of the center of bottom xy plane of a 3D box.

		rot_axis : list of tuple with length 3
			choose the rotation angle.

		angle	: float
			rotation angle as radian.

		RETURN
		------
		structure_list : list
			a list which has structure objects as its elements.
		"""

		xx = xycenter_of_bottom_3Dbox[0]
		yy = xycenter_of_bottom_3Dbox[1]
		zz = xycenter_of_bottom_3Dbox[2]

		structure_list = []

		for n in range(N):
			structure_list.append( structures.Box(mat, (xx+n*x_spacing-a/2, yy-b/2, zz),(xx+n*x_spacing+a/2, yy+b/2, zz+thick)))

		return structure_list

class Ushape(object):
	def __new__(self, mat, a,b,c, thick, center, rot_axis, angle):
		"""U shape split ring resonator.

				<-a->
				 ___		      ___	_ _
				|   |			 |	 |	 |
				|   |   center	 |	 |	 |
				|   |	   .	 |	 | 	 |
				|   |			 |	 |	 | c
				|   |____________|	 |	 |
				|					 |	 |
				|____________________|	_|_

				<-------- b --------->

		PARAMETERS
		----------
		mat :	material object
		a	:	float
		b	:	float
		c	: 	float
		thick	:	float
		center	:	list of tuple with length 3
		rot_axis:	list of tuple with length 3
		angle	:	float

		RETURNS
		-------
		Ushape : list

		"""

		x = center1[0]
		y = center1[1]
		z = center1[2]
	
		Ushape = []

		Ushape_row  = structures.Box( mat, (x-b/2  , y-c/2, z), (x+b/2  , y-c/2+a, z+thick), rot_axis, angle)
		Ushape_col1 = structures.Box( mat, (x+b/2-a, y-c/2, z), (x+b/2  , y+c/2  , z+thick), rot_axis, angle)
		Ushpae_col2 = structures.Box( mat, (x-b/2  , y-c/2, z), (x-b/2+a, y+c/2  , z+thick), rot_axis, angle)

		Ushape.append(Ushape_row)
		Ushape.append(Ushape_col1)
		Ushape.append(Ushape_col2)

		return Ushape

class U_up2_left2(object):
	def __new__(self, mat, a, b, c, thick, center1, center2, center3, center4, rot_axis, angle):
		"""Two left forks and Two upward forks.

					<-------- b -------->	<-a->
					 ___________________	 ___			 ___	_ _
					|					|	|   |			|	|	 |
					|_______________	|	|   |  center2	|	|	 |
									|	|	|	|	  		|	| 	 |
			center1	----->	  .		|	|	|	|	  .		|	|	 | c
					 _______________|	|	|	|___________|	|	 |
					|					|	|					|	 |
					|___________________|	|___________________|	_|_

					 ___		   	 ___	 ___________________
					|	|           |	|   |					|
					|	|  center3  |	|	|_______________	|
					|	|	 	    |	|					|	|
					|	|	  .	    |	|   center4   .		|	|
					|	|___________|	|	 _______________|	|
					|					|	|					|
					|___________________|	|___________________|

		PARAMETERS
		----------
		mat	:	material object
			an instance of material class which is declared in KEMP.material
		a	:	float
		b	:	float
		c	:	float
		thick	:	float
		center1	:	list or tuple with length 3
		center2	:	list or tuple with length 3
		center3	:	list or tuple with length 3
		center4	:	list or tuple with length 3
		rot_axis:	list or tuple with length 3
		angle	:	float

		"""

		x1 = center1[0]
		y1 = center1[1]
		z1 = center1[2]

		x2 = center2[0]
		y2 = center2[1]
		z2 = center2[2]

		x3 = center3[0]
		y3 = center3[1]
		z3 = center3[2]

		x4 = center4[0]
		y4 = center4[1]
		z4 = center4[2]

		structure_list = []

		left_fork1_column = structures.Box( mat, (x1-b/2, y1-c/2  , z1), (x1-b/2+a, y1+c/2  , z1+thick), rot_axis, angle)
		left_fork1_row1   = structures.Box( mat, (x1-b/2, y1+c/2+a, z1), (x1+b/2  , y1+c/2  , z1+thick), rot_axis, angle)
		left_fork1_row2   = structures.Box( mat, (x1-b/2, y1-c/2  , z1), (x1+b/2  , y1-c/2+a, z1+thick), rot_axis, angle)

		left_fork4_column = structures.Box( mat, (x4-b/2, y4-c/2  , z4), (x4-b/2+a, y4+c/2  , z4+thick), rot_axis, angle)
		left_fork4_row1   = structures.Box( mat, (x4-b/2, y4+c/2+a, z4), (x4+b/2  , y4+c/2  , z4+thick), rot_axis, angle)
		left_fork4_row2   = structures.Box( mat, (x4-b/2, y4-c/2  , z4), (x4+b/2  , y4-c/2+a, z4+thick), rot_axis, angle)

		up_fork2_row     = structures.Box( mat, (x2-b/2  , y2-c/2, z2), (x2+b/2  , y2-c/2+a, z2+thick), rot_axis, angle)
		up_fork2_column1 = structures.Box( mat, (x2+b/2-a, y2-c/2, z2), (x2+b/2  , y2+c/2  , z2+thick), rot_axis, angle)
		up_fork2_column2 = structures.Box( mat, (x2-b/2  , y2-c/2, z2), (x2-b/2+a, y2+c/2  , z2+thick), rot_axis, angle)

		up_fork3_row     = structures.Box( mat, (x3-b/2  , y3-c/2, z3), (x3+b/2  , y3-c/2+a, z3+thick), rot_axis, angle)
		up_fork3_column1 = structures.Box( mat, (x3+b/2-a, y3-c/2, z3), (x3+b/2  , y3+c/2  , z3+thick), rot_axis, angle)
		up_fork3_column2 = structures.Box( mat, (x3-b/2  , y3-c/2, z3), (x3-b/2+a, y3+c/2  , z3+thick), rot_axis, angle)

		structures_list.append(left_fork1_column)
		structures_list.append(left_fork1_row1)
		structures_list.append(left_fork1_row2)

		structures_list.append(left_fork4_column)
		structures_list.append(left_fork4_row1)
		structures_list.append(left_fork4_row2)

		structures_list.append(up_fork2_row)
		structures_list.append(up_fork2_column1)
		structures_list.append(up_fork2_column2)

		structures_list.append(up_fork3_row)
		structures_list.append(up_fork3_column1)
		structures_list.append(up_fork3_column2)

		return structure_list

class rhombus_shape(object):
	def __new__(self, a, b, rot_axis, rot_angle):
		
		structure_list = []

		return structure_list

class triangular_lattice(object):
	def __new__(self, a, b, rot_axis, rot_angle):
		
		structure_list = []

		return structure_list

class honeycomb_lattice(object):
	def __new__(self, a, b, rot_axis, rot_angle):
		
		structure_list = []

		return structure_list
