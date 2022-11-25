import pandas as pd
from sklearn.svm import SVR, NuSVR
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from matplotlib import pyplot as plt
# import plotly.express as px
from scipy.interpolate import griddata
import plotly.graph_objects as go
import dtreeviz.trees as dt
import pickle
# import umap
import shap
from lazypredict.Supervised import LazyRegressor
from xgboost import XGBRegressor
import xgboost as xgb
# from sklearn.metrics import accuracy_score
model_color = ["m", "c", "g"]
kernel_label = ["Linear", "RBF", "Polynomial"]

lw = 2
def adjusted_rsquared(r2, n, p):
	return 1 - (1 - r2) * ((n - 1) / (n - p - 1))

def score(model):
	return np.float32(adjusted_rsquared(model.score(X_test, y_test), X_test.shape[0], X_test.shape[1]))

def rmse(model):
	return mean_squared_error(y_test, model.predict(X_test), squared=False)

data = pd.read_csv('data/u_mf final data.csv').fillna(0)
# print(data)

# TODO: ADD SAUTER DIA and other equivalent dias representing the distr

X = data.drop(columns=['u_mf (m/s)', 'Dist Name'])
y = data['u_mf (m/s)']

best_adj_r2 = {'SVR':-np.inf, 'LR':-np.inf, 'NuSVR':-np.inf, 'DTR':-np.inf, 'XGB':-np.inf}
best_rmse = {'SVR':np.inf, 'LR':np.inf, 'NuSVR':np.inf, 'DTR':np.inf, 'XGB':np.inf}

for i in range(10):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

	# model_linear = SVR(kernel='linear', tol=1e-5)
	# model_linear.fit(X_train, y_train)

	model_linreg = LinearRegression()
	model_linreg.fit(X_train, y_train)

	model_nusvr = NuSVR(tol=1e-5)
	model_nusvr.fit(X_train, y_train)

	model_dtr = DecisionTreeRegressor(max_depth=3)
	model_dtr.fit(X_train, y_train)

	model_xgb = XGBRegressor(max_depth=3)
	model_xgb.fit(X_train, y_train)

	reg = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None, 
		regressors=[XGBRegressor, NuSVR, DecisionTreeRegressor, LinearRegression])
	scores,predictions = reg.fit(X_train, X_test, y_train, y_test)

	# print(scores)

	# [print(X.columns[i]+" : ",model.coef_[0][i]) for i in range(len(X.columns))]

	# svrs = [model_linear, model_rbf, model_poly, model_linreg, model_lasso, model_mlp]
	# svrs = [model_linear, model_linreg, model_nusvr, model_dtr, model_xgb]
	# for model in svrs:
		# print(model, 'R2: ', model.score(X_test, y_test), 'adj R2: ',
		# 	adjusted_rsquared(model.score(X_test, y_test), X_test.shape[0], X_test.shape[1]))
		# print(model.support_vectors_.shape)
	# print(model_xgb.n_estimators)
	# print(model_xgb.feature_importances_)
	# sorted_idx = np.argsort(model.feature_importances_)[::-1]
	# for index in sorted_idx:
	#     print([train.columns[index], model.feature_importances_[index]]) 
	# xgb.plot_importance(model_xgb)
	# plt.show()

	models = reg.provide_models(X_train, X_test, y_train, y_test)

	# for m in models:
	# 	print(m, score(models[m]), score(models[m]['regressor']))
	# viz = dt.dtreeviz(model_xgb, X_train, y_train, tree_index=0, target_name='u_mf', feature_names=list(X.columns), 
	# 	colors={'scatter_marker': '#ff84ac'})
	# viz.view()
	# print(adj_r2)

	# if score(model_linear) < score(models['SVR']):
	# 	model_linear = models['SVR']
	if score(model_linreg) < score(models['LinearRegression']):
		model_linreg = models['LinearRegression']
	if score(model_nusvr) < score(models['NuSVR']):
		model_nusvr = models['NuSVR']
	if score(model_dtr) < score(models['DecisionTreeRegressor']):
		model_dtr = models['DecisionTreeRegressor']
	if score(model_xgb) < score(models['XGBRegressor']):
		model_xgb = models['XGBRegressor']

	# if score(model_linear) > best_adj_r2['SVR']:
	# 	best_adj_r2['SVR'] = score(model_linear)
	# 	best_rmse['SVR'] = rmse(model_linear)
	# 	pickle.dump(model_linear, open('models/svr.mdl', 'wb'))

	if score(model_linreg) > best_adj_r2['LR']:
		best_adj_r2['LR'] = score(model_linreg)
		best_rmse['LR'] = rmse(model_linreg)
		pickle.dump(model_linreg, open('models/linear.mdl', 'wb'))

	if score(model_nusvr) > best_adj_r2['NuSVR']:
		best_adj_r2['NuSVR'] = score(model_nusvr)
		best_rmse['NuSVR'] = rmse(model_nusvr)
		pickle.dump(model_nusvr, open('models/nusvr.mdl', 'wb'))

	if score(model_dtr) > best_adj_r2['DTR']:
		best_adj_r2['DTR'] = score(model_dtr)
		best_rmse['DTR'] = rmse(model_dtr)
		pickle.dump(model_dtr, open('models/dtr.mdl', 'wb'))

	if score(model_xgb) > best_adj_r2['XGB']:
		best_adj_r2['XGB'] = score(model_xgb)
		best_rmse['XGB'] = rmse(model_xgb)
		pickle.dump(model_xgb, open('models/xgb.mdl', 'wb'))

	with np.printoptions(precision=4):
		print(i, best_adj_r2)

print('finally')
print(best_adj_r2)
print(best_rmse)
# xi = np.linspace(min(u_x_train[:,0]), max(u_x_train[:,0]), num=500)
# yi = np.linspace(min(u_x_train[:,1]), max(u_x_train[:,1]), num=500)
# x_grid, y_grid = np.meshgrid(xi, yi)
# z_pred_grid = griddata((u_x_train[:,0], u_x_train[:,1]), model_linear.predict(X_train), (x_grid, y_grid), method='cubic')
# z_true_grid = griddata((u_x_train[:,0], u_x_train[:,1]), y_train, (x_grid, y_grid), method='cubic')

# fig = go.Figure([go.Surface(x = x_grid, y = y_grid, z = z_pred_grid), go.Surface(x = x_grid, y = y_grid, z = z_true_grid)])
# fig.update_traces(contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True))
# fig.update_layout(title='Predictions UMAP',	scene_camera_eye=dict(x=1.87, y=0.88, z=-0.64), template = 'plotly_dark')

# fig.show()


# fig.update_layout(width=960, height=540, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
# 	title_text='Mortality distribution with Age', # title of plot
# 	# title_text='Severity distribution with Age', # title of plot
# 	xaxis_title_text='Age', # xaxis label
# 	yaxis_title_text='Count', # yaxis label
# 	bargap=0.2, # gap between bars of adjacent location coordinates
# 	bargroupgap=0.1, # gap between bars of the same location coordinates
# 	font = dict(size=22)
# )

def interpret_shap(model,df,name):
	# compute the SHAP values for every prediction in the dataset
	if isinstance(model, XGBRegressor):
		explainer = shap.TreeExplainer(model)
	else:
		explainer = shap.KernelExplainer(model.predict, df)
	shap_values = explainer.shap_values(df)
	f = plt.figure()
	ax1 = f.add_subplot(121)
	plt.title(name,fontsize=25)
	shap.summary_plot(shap_values, df, plot_type="bar", show=False)
	ax2 = f.add_subplot(122)
	shap.summary_plot(shap_values, df, show=False)
	ax1.set_xlabel('mean(|SHAP value|)\naverage impact on model output magnitude')
	ax1.xaxis.label.set_size(22)
	ax1.yaxis.label.set_size(22)
	ax1.tick_params(axis='both', labelsize=22)
	ax2.set_xlabel('SHAP value\nimpact on model output')
	ax2.xaxis.label.set_size(22)
	ax2.yaxis.label.set_size(22)
	ax2.tick_params(axis='both', labelsize=22)
	ax2.axes.yaxis.set_visible(False)
	plt.show()
	vals= np.abs(shap_values).mean(0)
	feature_importance = pd.DataFrame(list(zip(df.columns,vals)),columns=['col_name','feature_importance_vals'])
	feature_importance.sort_values(by=['feature_importance_vals'],ascending=False,inplace=True)
	print(feature_importance)