results = load( '/Applications/Fall18Courses/6.867/project/complex_model/resultsNP_complex.mat' );
results_np = results.results_np;
resonances = 40;
results_p = squeeze( results_np( :, 1, 1:resonances ) );
results_r = squeeze( results_np( :, 2, 1:resonances ) );
results_p_real = real( results_p );
results_p_im = imag( results_p );
results_r_real = real( results_r );
results_r_im = imag( results_r );

[ results, mu, sigma ] = zscore( [ results_p_real results_p_im results_r_real results_r_im ] );
save( '/Applications/Fall18Courses/6.867/project/complex_model/reformat_complex.mat', 'results' )
% k = load( '/Applications/Fall18Courses/6.867/project/study_ratio_sigma_spacing_to_spacing/reformat1.mat' );
samples = load( '/Applications/Fall18Courses/6.867/project/complex_model/test.mat' );
samples = samples.results_np;
conv_samples = zeros( size( samples ) );
for k = 1:size( samples, 2 )
   conv_samples( :, k ) = samples( :, k ) * sigma( k ) + mu( k );
end

results_sample = zeros( size( conv_samples, 1 ), 2, resonances );
results_sample( :, 1, : ) = conv_samples( :, 1:40 ) + 1i * conv_samples( :, 41:80 );
results_sample( :, 2, : ) = conv_samples( :, 81:120 ) + 1i * conv_samples( :, 121:160 );
save( '/Applications/Fall18Courses/6.867/project/complex_model/hidden_complex.mat', 'results_sample' )