%%intro signalbehandling Matlab 06.01.2015

close all; clear all;
% sinus/ cosinus signal
% x(t)=A*sin(2pi*f1*t+ fase)
fs=8000;
f1=200;
t=-0.05:1/fs:0.005;
A=1;
% x[n]=-->xn
xn=A*sin(2*pi*f1*t);
figure(1)
subplot(3,1,1)
plot(t,xn); hold on
stem(t,xn,'.r'); hold off
xlabel('Tid (sek)');
ylabel('x[n]');
title('x(t)=A*sin(2*pi*f1*t)')
% hvorfor er sinus signal intressant?
% Fourier rekke.
% Er viktig fordi vi kan bryte et signal ned i sinus signal med ulik
% frekvens, fase og amplitude.

% Trekant pulstog
xn1=A*sawtooth(2*pi*f1*t);
% Firekant pulstog
xn2=A*square(2*pi*f1*t);
subplot(3,1,2)
plot(t,xn1); hold on
stem(t, xn1, '.r'); hold off
% støy (noise)
xn3=0.1*randn(size(t));
figure(2)
subplot(3,1,1)
plot(t,xn3); hold on
stem(t,xn3,'.r'); hold off
title('støy signal')
xn4=xn+xn3;
subplot(3,1,2)
plot(t, xn4); hold on
stem(t, xn4, '.r'); hold off
title('signal+ støy signal')

% >> SPTOOL
% Har sett på frekvens spekter av sinus signalet (FFT, 1024)
% Ser at f=200Hz stemmer.
% Såg også på tågfløyta litt meir komplekst signal.

% Spektrogram

% Lager eit sammensatt signal
xn5=[xn xn1 xn2 xn3 xn4];
f=0:0.5:fs/2;
figure(3)
spectrogram(xn5,256,128,f,fs,'yaxis')
