plot( losses8( :, 1 ), losses8( :, 3 ), 'b', 'Linewidth', 2 )
hold on
plot( losses16( :, 1 ), losses16( :, 3 ), 'r', 'Linewidth', 2 )
plot( losses64( :, 1 ), losses64( :, 3 ), 'k', 'Linewidth', 2 )
legend( '8 hidden units', '16 hidden units', '64 hidden units' )
xlabel('Epochs')
ylabel('Reconstruction Error')
title('Reconstruction Error (Hidden Units)')
ax = axis;
axis( [ ax(1) ax(2) 0 ax(4) ] )
set( gca, 'Fontsize', 14 )
hold off

figure
plot( losses8( :, 1 ), losses8( :, 4 ), 'b', 'Linewidth', 2 )
hold on
plot( losses16( :, 1 ), losses16( :, 4 ), 'r', 'Linewidth', 2 )
plot( losses64( :, 1 ), losses64( :, 4 ), 'k', 'Linewidth', 2 )
legend( '8 hidden units', '16 hidden units', '64 hidden units' )
xlabel('Epochs')
ylabel('Latent Error')
title('Latent Error (Hidden Units)')
ax = axis;
axis( [ ax(1) ax(2) 0 ax(4) ] )
set( gca, 'Fontsize', 14 )
hold off

figure
plot( losses0( :, 1 ), losses0( :, 3 ), 'b', 'Linewidth', 2 )
hold on
plot( losses1( :, 1 ), losses1( :, 3 ), 'r', 'Linewidth', 2 )
plot( losses2( :, 1 ), losses2( :, 3 ), 'k', 'Linewidth', 2 )
legend( '0 extra hidden layers', '1 extra hidden layer', '2 extra hidden layers' )
xlabel('Epochs')
ylabel('Reconstruction Error')
title('Reconstruction Error (Hidden Layers)')
ax = axis;
axis( [ ax(1) ax(2) 0 ax(4) ] )
set( gca, 'Fontsize', 14 )
hold off

figure
plot( losses0( :, 1 ), losses0( :, 4 ), 'b', 'Linewidth', 2 )
hold on
plot( losses1( :, 1 ), losses1( :, 4 ), 'r', 'Linewidth', 2 )
plot( losses2( :, 1 ), losses2( :, 4 ), 'k', 'Linewidth', 2 )
legend( '0 extra hidden layers', '1 extra hidden layer', '2 extra hidden layers' )
xlabel('Epochs')
ylabel('Latent Error')
title('Latent Error (Hidden Layers)')
ax = axis;
axis( [ ax(1) ax(2) 0 ax(4) ] )
set( gca, 'Fontsize', 14 )
hold off

figure
plot( losses_lrelu( :, 1 ), losses_lrelu( :, 3 ), 'b', 'Linewidth', 2 )
hold on
plot( losses_relu( :, 1 ), losses_relu( :, 3 ), 'r', 'Linewidth', 2 )
plot( losses_sig( :, 1 ), losses_sig( :, 3 ), 'k', 'Linewidth', 2 )
legend( 'Leaky ReLU', 'ReLU', 'Sigmoid' )
xlabel('Epochs')
ylabel('Reconstruction Error')
title('Reconstruction Error (Activation Function)')
ax = axis;
axis( [ ax(1) ax(2) 0 ax(4) ] )
set( gca, 'Fontsize', 14 )
hold off

figure
plot( losses_lrelu( :, 1 ), losses_lrelu( :, 4 ), 'b', 'Linewidth', 2 )
hold on
plot( losses_relu( :, 1 ), losses_relu( :, 4 ), 'r', 'Linewidth', 2 )
plot( losses_sig( :, 1 ), losses_sig( :, 4 ), 'k', 'Linewidth', 2 )
legend( 'Leaky ReLU', 'ReLU', 'Sigmoid' )
xlabel('Epochs')
ylabel('Latent Error')
title('Latent Error (Activation Function)')
ax = axis;
axis( [ ax(1) ax(2) 0 ax(4) ] )
set( gca, 'Fontsize', 14 )
hold off