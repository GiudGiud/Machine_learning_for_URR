resonances = 40;
results_p = squeeze( results_np( :, 1, 1:resonances ) );
results_r = squeeze( results_np( :, 2, 1:resonances ) );
results_p_real = real( results_p );
results_p_im = imag( results_p );
results_r_real = real( results_r );
results_r_im = imag( results_r );

results = transpose( [ results_p_real results_p_im results_r_real results_r_im ] );