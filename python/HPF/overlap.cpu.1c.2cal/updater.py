def Hzupdater(QBz,QHz):

	if self.rank == 1:

		zz = self.EEHgrid_per_node[2]

		for k in range(zz-1):

			previous_k = self.Bz[:,:,k].copy()

			CBz1_k = self.CBz1[:,:,k]
			CBz2_k = self.CBz2[:,:,k]
			CHz1_k = self.CHz1[:,:,k]
			CHz2_k = self.CHz2[:,:,k]
			CHz3_k = self.CHz3[:,:,k]

			diffxEy_k = ift(self.ikx * ft(self.Ey[:,:,k], axes=(0,)), axes=(0,))
			diffyEx_k = ift(self.iky * ft(self.Ex[:,:,k], axes=(1,)), axes=(1,))
			self.Bz[:,:,k] = CBz1_k * self.Bz[:,:,k] + CBz2_k * (diffxEy_k - diffyEx_k)
			self.Hz[:,:,k] = CHz1_k * self.Hz[:,:,k] + CHz2_k * self.Bz[:,:,k] + CHz3_k * previous_k

		QBz.put(self.Bz[:,:,0:-1])
		QHz.put(self.Hz[:,:,0:-1])

	elif self.rank > 1 and self.rank < (self.size-1):

		zz = self.EEHgrid_per_node[2]

		for k in range(zz-1):

			previous_k = self.Bz[:,:,k].copy()

			CBz1_k = self.CBz1[:,:,k]
			CBz2_k = self.CBz2[:,:,k]
			CHz1_k = self.CHz1[:,:,k]
			CHz2_k = self.CHz2[:,:,k]
			CHz3_k = self.CHz3[:,:,k]

			diffxEy_k = ift(self.ikx * ft(self.Ey[:,:,k], axes=(0,)), axes=(0,))
			diffyEx_k = ift(self.iky * ft(self.Ex[:,:,k], axes=(1,)), axes=(1,))
			self.Bz[:,:,k] = CBz1_k * self.Bz[:,:,k] + CBz2_k * (diffxEy_k - diffyEx_k)
			self.Hz[:,:,k] = CHz1_k * self.Hz[:,:,k] + CHz2_k * self.Bz[:,:,k] + CHz3_k * previous_k

		QBz.put(self.Bz[:,:,0:-1])
		QHz.put(self.Hz[:,:,0:-1])

	elif self.rank == (self.size-1):

		zz = self.EEHgrid_per_node[2]

		for k in range(zz):

			previous_k = self.Bz[:,:,k].copy()

			CBz1_k = self.CBz1[:,:,k]
			CBz2_k = self.CBz2[:,:,k]
			CHz1_k = self.CHz1[:,:,k]
			CHz2_k = self.CHz2[:,:,k]
			CHz3_k = self.CHz3[:,:,k]

			diffxEy_k = ift(self.ikx * ft(self.Ey[:,:,k], axes=(0,)), axes=(0,))
			diffyEx_k = ift(self.iky * ft(self.Ex[:,:,k], axes=(1,)), axes=(1,))
			self.Bz[:,:,k] = CBz1_k * self.Bz[:,:,k] + CBz2_k * (diffxEy_k- diffyEx_k)
			self.Hz[:,:,k] = CHz1_k * self.Hz[:,:,k] + CHz2_k * self.Bz[:,:,k] + CHz3_k * previous_k

		QBz.put(self.Bz[:,:,:])
		QHz.put(self.Hz[:,:,:])

def Ezupdater():

	if self.rank == 1:

		zz = self.HHEgrid_per_node[2]

		for k in range(zz):

			previous_k = self.Dz[:,:,k].copy()

			CDz1_k = self.CDz1[:,:,k]
			CDz2_k = self.CDz2[:,:,k]
			CEz1_k = self.CEz1[:,:,k]
			CEz2_k = self.CEz2[:,:,k]
			CEz3_k = self.CEz3[:,:,k]

			diffxHy_k = ift(self.ikx * ft(self.Hy[:,:,k], axes=(0,)), axes=(0,))
			diffyHx_k = ift(self.iky * ft(self.Hx[:,:,k], axes=(1,)), axes=(1,))
			self.Dz[:,:,k] = CDz1_k * self.Dz[:,:,k] + CDz2_k * (diffxHy_k - diffyHx_k)
			self.Ez[:,:,k] = CEz1_k * self.Ez[:,:,k] + CEz2_k * self.Dz[:,:,k] + CEz3_k * previous_k

		QDz.put(self.Dz[:,:,:])
		QEz.put(self.Ez[:,:,:])

	elif self.rank > 1 and self.rank < (self.size-1):

		zz = self.HHEgrid_per_node[2]

		for k in range(zz-1):

			previous_k = self.Dz[:,:,k].copy()

			CDz1_k = self.CDz1[:,:,k]
			CDz2_k = self.CDz2[:,:,k]
			CEz1_k = self.CEz1[:,:,k]
			CEz2_k = self.CEz2[:,:,k]
			CEz3_k = self.CEz3[:,:,k]

			diffxHy_k = ift(self.ikx * ft(self.Hy[:,:,k], axes=(0,)), axes=(0,))
			diffyHx_k = ift(self.iky * ft(self.Hx[:,:,k], axes=(1,)), axes=(1,))
			self.Dz[:,:,k] = CDz1_k * self.Dz[:,:,k] + CDz2_k * (diffxHy_k - diffyHx_k)
			self.Ez[:,:,k] = CEz1_k * self.Ez[:,:,k] + CEz2_k * self.Dz[:,:,k] + CEz3_k * previous_k

		QDz.put(self.Dz[:,:,1:])
		QEz.put(self.Ez[:,:,1:])

	elif self.rank == (self.size-1):

		zz = self.HHEgrid_per_node[2]

		for k in range(zz-1):

			previous_k = self.Dz[:,:,k].copy()

			CDz1_k = self.CDz1[:,:,k]
			CDz2_k = self.CDz2[:,:,k]
			CEz1_k = self.CEz1[:,:,k]
			CEz2_k = self.CEz2[:,:,k]
			CEz3_k = self.CEz3[:,:,k]

			diffxHy_k = ift(self.ikx * ft(self.Hy[:,:,k], axes=(0,)), axes=(0,))
			diffyHx_k = ift(self.iky * ft(self.Hx[:,:,k], axes=(1,)), axes=(1,))
			self.Dz[:,:,k] = CDz1_k * self.Dz[:,:,k] + CDz2_k * (diffxHy_k - diffyHx_k)
			self.Ez[:,:,k] = CEz1_k * self.Ez[:,:,k] + CEz2_k * self.Dz[:,:,k] + CEz3_k * previous_k

		QDz.put(self.Dz[:,:,1:])
		QEz.put(self.Ez[:,:,1:])
