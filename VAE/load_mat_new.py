import scipy.io

mat = scipy.io.loadmat('/Applications/Fall18Courses/6.867/project/complex_model/complex_res_data_20r_1000sa_5sp_0.4w.mat')
scipy.io.savemat('/Applications/Fall18Courses/6.867/project/complex_model/resultsNP_complex.mat', mdict = {'results_np': mat['results_np']})
