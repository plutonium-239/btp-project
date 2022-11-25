import numpy as np
import plotly.express as px
import pandas as pd

rng = np.random.default_rng()

def bubble_diameter():
	n_samples = 1
	n_points = 100
	u_mf = rng.normal(1.8, 0.7, n_samples) # 0.5 - 20 cm/s
	# u_mf = np.array([1.8, 1.8])
	u_mf[u_mf < 0.5] = 0.5
	u_mf[u_mf > 20] = 20
	u_0 = u_mf + rng.normal(24, 24/2, n_samples) # cm/s
	# u_0 = np.array([9,9])
	u_0[u_0 < u_mf] = 10
	D_t = rng.uniform(20, 500, n_samples) # diameter in cm, A_t = pi*D_t^2
	# D_t = np.array([20,100])
	A_t = np.pi*(D_t**2) # cm^2
	nd = 3000

	D_b0 = 0.347*np.power((A_t*(u_0 - u_mf)/nd), 2/5) # 
	D_bM = 0.347*1.87*np.power((A_t*(u_0 - u_mf)), 2/5)
	d_p =  rng.normal(25.5, 15.6)/1000 # particle diameter (0.006-0.0045)

	h = rng.uniform(0, 30, n_points)
	corrs = ['moriwen', 'park', 'whitehead', 'geldart']
	vals = pd.DataFrame(columns=['height']+corrs)
	vals.loc[:, 'height'] = h

	conditions = []
	for i in range(n_samples):
		# Mori and Wen
		d_b_moriwen = D_bM[i] - (D_bM[i] - D_b0[i])*np.exp(-0.3*h/D_t[i])
		print(d_b_moriwen.mean())
		istr = f"u_mf={u_mf[i]:.2f}, u_0={u_0[i]:.2f}, D_t={D_t[i]:.2f}, D_b0={D_b0[i]:.2f}, D_bM={D_bM[i]:.2f}"
		# d_b += rng.normal(0,0.02*d_b.mean(), n_points)
		print(istr)
		conditions.append(istr)

		# Park et al. 
		d_b_park = 33.3*(d_p**1.5)*(u_0/u_mf - 1)**0.77 *h

		# Whitehead et al.
		d_b_wht = 9.76*(u_0/u_mf)**(0.33*(0.032*h)**0.54)

		# Geldart
		d_b_gld = 0.027*(u_0 - u_mf)**0.94 *h

		vals.loc[:, 'moriwen'] = d_b_moriwen
		vals.loc[:, 'park'] = d_b_park
		vals.loc[:, 'whitehead'] = d_b_wht
		vals.loc[:, 'geldart'] = d_b_gld

	fig = px.scatter(vals, x='height', y=corrs, title="Bubble Size correlation", trendline='lowess')
	fig.update_layout(
		yaxis_title='Bubble Size distribution (cm)', xaxis_title = 'height (cm)', legend_title='Runs', font=dict(size=22), width=1200, height=750)
	# fig.show()
	fig.update_traces(line=dict(dash="dash"))
	fig.show()



bubble_diameter()