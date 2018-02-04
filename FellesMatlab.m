%% Sinus signal med 2 frekvenser + støy, filtrering, frekvensanalyse 090115+16012015
% Lager filteret i >>> FDAtool
close all; clear all;
f1 = 100;   f2 = 200;
fs = 10000;
A1 = 5;     A2 = 2.5;
t = 0:1/fs:2;
xn1 = A1*sin(2*pi*f1*t);   
xn2 = A2*sin(2*pi*f2*t);
xn=xn1+xn2;
% xn = [xn1 xn2 xn3 ...]
xnn = xn+rand(size(t));
figure(1)
subplot(2,1,1)
plot(t,xnn); hold on
plot(t(10000:10500),xnn(10000:10500),'r'); hold off
subplot(2,1,2)
plot(t(10000:10500),xnn(10000:10500),'r'); hold off
axis([1.0 1.05 -15 15])

% Lager digitalt filter i FDAtool
% Spec(Spesifikasjon): 
% LP -type fpass = 120Hz, fstopp = 190Hz, demping = 69dB, rippel = 0.1dB
load C:\Users\Bruker\Documents\MATLAB\Filterkoeff1.mat;
b=Num;
% y[n] = b0*x[n]+b1*x[n-1]+b2*x[n-2]+...+b388*x[n-388];
% b = [b0 b1 b2 ... b388]

% Filtrering
yn = filter(b,1,xnn);
figure(2)
subplot(2,1,1)
plot(t(10000:10500),xnn(10000:10500),'r'); hold on
plot(t(10000:10500),yn(10000:10500),'b'); hold off
axis([1.0 1.05 -15 15])
xlabel('Tid(s)')
title('Signal xnn = A1*sin(2*pi*f1*t)+A2*sin(2*pi*f1*t) + støy')
legend('Signal før filter', 'Signal etter filter')
grid on

% Forts, 16.01.2015
% Frekvensanalyse:
% Tar FFT av signalet før og etter filteret
% FFT-> synonymt med frekvens analyse.
% N punkt FFT- Fast Fourier Transform
% N = 2^n
% N = 2048, 1024, 512, ...
N = 2048;
% x[k]-> Xk
Xk = fft(xn,N);
XK = abs(Xk);
Yk = fft(yn,N);
YK = abs(Yk);
% skalerer til frekvens
f = 0:fs/N:fs*(N-1)/N;

figure(3)
%plot(f(1:200),XK(1:200),'r'); hold on
%plot(f(1:200),YK(1:200),'b'); hold off
plot(f,XK,'r');
%plot(f,);
grid on
xlabel('Frekvens (Hz)')
title('Frekvens Spekter')
% dersom du vil syn støygolvet (sterke og svake komponenter 
% samtidig) -> 20*log10(XK)

figure(4)
XKK = 20*log10(XK);
YKK = 20*log10(YK);
plot(f(1:200),XKK(1:200),'r'); hold on
plot(f(1:200),YKK(1:200),'b'); hold off
grid on
xlabel('Frekvens (Hz)')
title('Frekvens Spekter')
ylabel('dB spekter')
%kommentar 1:
% Størrelsen på FFT'n bestemmer frekvensoppløsninga.
% Dette er bredda på peak'n i frekvens spekteret
% Kommentar 2:
% Det er viktig å forstå når du bruker FFT så tar du ut kun de første N
% samplene i signalfilen og tar FFT av dem.
% Utsnittet heter segment(blokk).
% xn(N:2*N-1)

%% ECG data 19012015
close all; clear all;
load C:\Users\Bruker\Documents\MATLAB\sample_ecg.mat
t= ecg(:,1);
xn = ecg(:,2);
figure(1)
plot(t,xn)
title('ECG data')
xlabel('Tid(sek)')
% sampletida
Ts = t(2)-t(1);
%fs = 1/Ts;

% 1. >>sptool-> signalbrowser foreta målinger
% 2. sptool filter design
% Type= LPF Low Pass Filter
% equrippel
% fpass = 10Hz
% fstopp = 20Hz
% Apass = 0.1dB (rippel)
% Astopp = 80dB (1/10000)
% Antal koeff N = 85;
%
% Konklusjon: Vi ser at filtrerte data er forskjøve/forsinka(delay)
%      Forsinkinga er N/2 eller (N-1)/2
%      (85-1)/2 = 42 sampels

%% Bildematrise 19012015
close all; clear all;
% lager testmatrise
I= [0 0 0 0 0 0 0 0 0 0 0;
    0 0 0 0 0 1 0 0 0 0 0;
    0 0 0 0 1 2 1 0 0 0 0;
    0 0 0 1 2 3 2 1 0 0 0;
    0 0 1 2 3 4 3 2 1 0 0;
    0 1 2 3 4 5 4 3 2 1 0;
    1 2 3 4 5 6 5 4 3 2 1];
    
    
figure(1)
imshow(I)
figure(2)
imshow(I,[])
figure(3)
imshow(I,[], 'InitialMagnification','fit')
impixelinfo
% imshow(I,[]) finner max og min i matrisa max=6-> hvit, min=0->svart
% normalt 0-> svart og 255-> hvit

I1= ones(7,11)*6;
I2=I1-I;
figure(4)
imshow(I2,[], 'InitialMagnification','fit')
impixelinfo

% leser inn et bilde, testbilde fra Matlab cameraman
I3=imread('cameraman.tif');
figure(5)
imshow(I3,[])
impixelinfo

% tar ut profil rekke 80(horisontalt)
% går ifra 2D til -> 1D signal.
xn80= I3(80,:);
figure(6)
subplot(2,1,1)
plot(xn80,'r')
figure(7)
imshow(I3,[]); hold on
plot(1:256,80*ones(1,256),'r','linewidt',1); hold off
impixelinfo

%% 200115
%close all; clear all;
% y[n]=0.5*y(n]-0.3*x[n-1];
a=1;
b=[0.50 -0.30];
n=0:1:19;
xn=sin(2*pi*n/9);
yn=filter(b,a,xn);
figure(1)
stem(n,xn,'.b'); hold on
stem(n,yn,'.r'); hold off
legend('x[n]-innsignal','y[n]- utsignal')
xlabel('Sample number')
title('y[n]=0.5*y(n]-0.3*x[n-1];')
% Hva gjør algoritmen (type filter)?
% >>fvtool(b,a)
% Vi ser at dette er et HP(High Pass) filter, men et dårlig filter,
% pga steilheten er dårlig.
% "Gode filter har 200- 300 koeffisienter"


% y[n]=0.50*y[n-1]+x[n]
% a=[1 -0.5]
% b=1
% fvtool(1,[1 -0.50])

% Nye algoritmer
% 1. y[n]=x[n]+x[n-1];
% 2. y[n]=x[n]-x[n-1];

% fvtool([1 1],1,[1 -1],1)

% Konklusjon: Algoritme 1 -> LPF; Algoritme 2 -> HPF.
a1=[1];
b1=[1 1];
yn1=filter(b1,a1,xn);

figure(2)
stem(n,xn,'.b'); hold on
stem(n,yn,'.r')
stem(n,yn1,'.k')
plot(n,yn1,'.k')
hold off
legend('x[n]-innsignal','y[n]- utsignal','y1[n] - utsignal')
xlabel('Sample number')
title('y[n]=0.5*y(n]-0.3*x[n-1], y[n]=x[n]+x[n-1]')


%ny algoritme:
% y[n]=(1/M)*(x[n]+x[n-1]+x[n-2+...+x[n-M]);
% y[n]=1/2*(x[n]+x[n-1])
M=100;
b=ones(1,M)*(1/M);
a=1;
fvtool(b,a,[1 1]*1/2,1,b2,1)
%% Var ikke i timen!

%% Filtrering Intro Labøving3 300115

% 4 metoder for å utføre filterering
close all; clear all;
%lager test sekvens xn-->x[n]
n=0:1:40;
xn=(1.20).^n +0.50*sin((2*pi*n/8)+pi/2);
figure(1)
subplot(3,1,1)
stem(n,xn)
axis([0 50 0 3])
% justerer testsignalet med å sette på nuller foran og bak.
xx=[zeros(1,5) xn zeros(1,10)];
n=0:1:length(xx)-1;
subplot(3,1,2)
stem(n,xx)
axis([0 50 0 3])
%setter to sample galt ('spike')
xns=xx;
xns(22)=2.5; xns(27)=0;
subplot(3,1,3)
stem(n,xns)
axis([0 50 0 3])

% Algoritme
% y[n]= (1/3)x[n]+(1/3)x[x-1]+(1/3)x[n-2]
% FIR type
b=[1/3 1/3 1/3];
a=(1);

% METODE 1 Folding
yn=conv(b,xns);

figure(2)
subplot(3,1,1)
stem(n,xns)
axis([0 50 0 2000])
subplot(3,1,2)
n1=0:1:length(yn)-1;
stem(n1,yn)
axis([0 50 0 2000])

% observerer start/slutt problem, forsinking (M/2) eller (M-1/2)
% (3-1)/2 = sample
% ser også at algoritma glatter dataene

% METODE 2 yn= filter(b,a,xn)
% bilde filter2()
yn1=filter(b,a,xns);
subplot(3,1,3)
n2=0:1:length(yn1)-1;
stem(n2,yn1)
axis([0 50 0 2000])

% METODE 3 Dobbelfiltrering yn= filtfilt(b,a,xn)
yn2=filtfilt(b,a,xns);
figure(3)
subplot(3,1,1)
stem(n,xns)
axis([0 50 0 3])
subplot(3,1,2)
n3=0:1:length(yn2)-1;
stem(n3,yn2)
axis([0 50 0 3])

% METODE 4
% x[n]
% x[n-1]=xn1
% x[n-2]=xn2
xn1=0;  xn2=0;
for i=1:1:length(xns)
    xn=xns(i);
    yn(i)=(1/3)*(xn+xn1+xn2);
    xn2=xn1;
    xn1=xn;
end
subplot(3,1,3)
n4=0:1:length(yn)-1;
stem(n4,yn)
axis([0 50 0 3])

%% Bildebehandling intro lab 3 06022015
close all; clear all;
% Testbilde baboon.mat
% Tester ut filtrering FIR-type algoritme
% 1D--> rekkevis/ kollonevis prosessering
% (horizontalt/vertikalt)
load baboon.mat
I = xx;
figure(1)
imshow(I,[]); hold on
% imshow(I,[]) finner min og max i bildematrisa
% min -> svart, max -> hvit
axis on
xnh100= I(100,:);
plot((1:256),100*ones(1,256),'r','linewidth',2); hold off

figure(2)
subplot(2,1,1)
plot(xnh100,'r')
% glattefilter moving average filter
b1 = (1/5)*ones(1,5);
b2 = (1/20)*ones(1,20);
ynh100=conv(b1,xnh100);
ynh100=conv(b2,xnh100);
subplot(2,1,2)
plot(ynh100,'r'); hold on
plot(ynh100,'k'); hold off
% Observerer at profilen blir glatta, mest glatting med 
% flest koeff. Forskyvning (delay)

% Filtrerer heile bilde.
Ih=conv2(I,b2);
figure(3)
imshow(Ih,[])
% bilde er blitt større horisontalt
% N+M-1 = 256+20-1= 275.
% Dei 19 første pikselene er gale og dei 19 siste pikselene er gale
% resulterer i svart "sørgerand" .

 
% Filtrerer så bilde vertikalt

Iv = conv2(I,b2');
figure(4)
imshow(Iv,[])
Ihv = conv2(Ih,b2');

figure(5)
imshow(Ihv,[])
%II=I(19:256,19:256)
% tar ut de delene av matrisa som ikke har sørgerand

% Bildebehandling med masker:
% m1 -> LP maske ("blurring")
M1 = [1 1 1;
      1 1 1;
      1 1 1;]*(1/9);
M2 = ones(5,5)*(1/25);
M3 = [-1 -1 -1;
      -1 8 -1;
      -1 -1 -1;];
M4 = [1 1 1;
      1 -8 1;
      1 1 1;];

IM2 = conv2(I,M2,'valid');
figure(6)
imshow(IM2,[])

IM3 = conv2(I,M3,'valid');
figure(7)
imshow(IM3,[])

IM4 = conv2(I,M4,'valid');
figure(8)
imshow(IM4,[])

% M = fspecial('gaussian', [5 5])
% fspecial har forskjellige filter som vi kan bruke

%% Delefilter 10022015 

close all; clear all;

% y[n] = x[n]-x[n-1]
% y[n] = x[n]+x[n-1]
% 2. Derivert
% y[n] = x[n]-2x[n-1]+x[n-2]
% 4. Deriverte
% H(z) = 1-4z^(-1)+6z^(-2)-4z^(-3)+z^(-4)
b1 = [1 -1];
b2 = [1 1];
b3 = [1 -2 1];
b4 = [1 -4 6 -4 1];
fvtool(b1,1,b2,1,b3,1,b4,1);

%% Polar og nullpunkt 13022015

close all; clear all;

%**************************************
% Eksperiment 2:
% H(z) = 1/(1+1.1z^(-1))
% H(z) = z/(z+0.90)
b = (1);
a = [1 1.1];
figure(3)
zplane(b,a)
title('pol- nullpunkt kart')
figure(4)
% impulseresponse h[n]
impz(b,a,40)

%**************************************
% Eksperiment 1:
% y[n] = x[n]-0.90y[n-1]
% H(z) = 1/(1+10.90z^(-1))
% H(z) =z/(z+0.90)
b = 1;
a = [1 0.90];
figure(3)
zplane(b,a)
title('pol- nullpunkt kart')
figure(4)
% impulseresponse h[n]
impz(b,a,40)

%**************************************
% Eksperiment 3:
% zn1 = zn2 = 0, 2 nullpunkt i origo
% komplekskonjugerte polar
% zp1 = 0.70exp(j*pi/3);
% zp2 = 0.70exp(-j*pi/3)
%
% H(z) = (((z+zn1)(z+zn2))/((z+zp1)(z+zp2)))

zn = [0 0];
zp = [0.70*exp(1i*pi/3) 0.70*exp(-1i*pi/3)];
figure(5)
zplane(zn',zp')
b = 1;
a = poly(zp);
zplane(b,a)

figure(6)
impz(b,a)

%*****************************
% Eksperiment 5:
% H(z) = (z^6+z^5+ z^4+z^3+z^2+z^1+1)/
%        (z^6+3z^5+(121/30)z^4+(92/30)z^3)
%
b = [1 1 1 1 1 1 1];
a = [1 3 121/30 92/30 41/30 1/3 1/30];
[zn,zp,k] = tf2zp(b,a);
figure(7)
zplane(b,a)
%zplane(zn,zp)
% zn,zp er kolonne vektor

% Bruker FDATool 

% Designe FIR-type filter. Observerte at alle polar låg i origo.
% Det er alltid slik for FIR- type filter.
% Nullpunkta låg både innenfor og utenfor enhetssirkelen.
% Ser også at mange nullpunkt ligger som perler på ei snor 

%% Frekvensrespons 16022015
close all; clear all;
% Algoritme: y[n]-0.40y[n-1]=x[n]
% Transferfunksjon: H(z)= 1/(1-0.40z^(-1))=z/(z-0.40)
% Sekvens x[n]=0.50*cos ((pi/4)*n)
n = 0:1:50;
xn = 0.50*cos((pi/4)*n)+1.5*ones(1,length(n));
b = [1];
a = [1 -0.40];
load Filterkoeff01.mat;
yn1 = filter(b1,a1,xn);
% Kan ikkje bruke conv() pga rekursiv algoritme
yn = filter(b,a,xn);
figure(1)
subplot(2,1,1)
stem(xn,'.b'); hold on
stem(yn,'.r'); hold off

subplot(2,1,2)
stem(xn,'.b'); hold on
stem(yn1,'.r'); hold off

% Frekvensresponsen |H(f)| (tallverdi(magnitude))
[H,w]=freqz(b,a);
figure(3)
plot(w/pi,abs(H))
grid on
xlabel('Normalisert frekvens')
ylabel('|H(f)|')
title('frekvensrespons')
% Regner ut eksakt verdi for w= 0(w=2*pi*f), w=pi/4.
H = freqz(b,a,[0 pi/4]);
HH = abs(H);
% HH = [1.6667    1.2972]
% DC(w=0) => 1.50*1.6667=2.490;
% w=pi/4 => 0.50*1.2972=0.6450;

% brukte fdatool og så at delay i filteret 
% for frekvensen fs/4 var ca. -21 grader

% lagde nytt filter i FDAtool interaktivt. Plasser 2 komplekskonjugerte 
% nullpunkt på enhetssirkelen ved w = pi/4 (f= fs/8)
% Konklusjon: Filter fjerner frekvensen helt.

% Observerer også godt startsproblem


%% Pole nullpunkt kart Frekvensrespons 17022015
close all; clear all;
% Eksperiment 1
% y[n] = 1/N(x[n]+x[n-1]+...+x[n-N])
N = 10;
b = ones(1,N)*(1/N); b2 = ones(1,2*N)*(1/(2*N));
a = 1;
[H,w] = freqz(b,a);
[H2,w] = freqz(b2,a);
figure(1)
plot(w/pi,abs(H)); hold on
plot(w/pi,abs(H2),'r'); hold off
figure(2)
zplane(b,a)
figure(3)
zplane(b2,a)

% konklusjon: Nullpunkt på enhetssirkelen tvinger frekvensresponsen lik 0.
%             Ved den frekvensen.

% Eksperiment 2:
p1 = 0.90*exp(1i*pi/4);
p = [p1 p1']';
n = [1 -1]';
[b,a]= zp2tf(n,p,1);
[H,w]= freqz(b,a);
figure(4)
plot(w/pi,abs(H)/max(abs(H)));
figure(5)
zplane(b,a)
% konklusjon: kraftig peak i |H(f)| der polen ligger nær enhetssirkelen 
% |H(f)| = 0 der vi har nullpunkt på enhetssirkelen.

% Eksperiment 3:
p1 = 0.90*exp(1i*pi/4);
n2 = 1.0*exp(1i*pi/8);
n3 = 1.0*exp(1i*3*pi/8);
pp = [0 0 0 0 p1 p1']';
nn = [1 -1 n2 n2' n3 n3']';

% legger inn tre polar på samme plass (komplekskojugert)
p = [0 0 0 0 p1 p1' p1 p1' p1 p1']';
n = [1 -1 n2 n2' n3 n3']';

[b,a]= zp2tf(nn,pp,1);
[b1,a1] = zp2tf(n,p,1);
[H,w]= freqz(b,a);
[H1,w]= freqz(b1,a1);
figure(6)
plot(w/pi,abs(H)/max(abs(H))); hold on
plot(w/pi,abs(H1)/max(abs(H1)),'r'); hold off
figure(7)
zplane(b,a)
figure(8)
zplane(b1,a1)

% Konklusjon: Flere polar på samme plass fører til skarpere (steilere)
%             filter.

% Eksperiment 4:
p1 = 0.90*exp(1i*pi/4);
p2 = 0.90*exp(1i*(pi/4+pi/20));
p3 = 0.90*exp(1i*(pi/4-pi/20));
n2 = 1.0*exp(1i*pi/8);
n3 = 1.0*exp(1i*3*pi/8);
p = [p1 p1' p2 p2' p3 p3']';
n = [1 -1 n2 n2' n3 n3']';

[b2,a2] = zp2tf(n,p,1);
[H2,w]= freqz(b2,a2);
[H1,w]= freqz(b1,a1);
figure(9)
plot(w/pi,abs(H)/max(abs(H))); hold on
plot(w/pi,abs(H1)/max(abs(H1)),'r'); hold off
figure(10)
zplane(b2,a2)

% konklusjon: "cluster" of polar i nærheten of hverandre ute ved
%             enhetssirkelen gir breidare "peak"


%% Bildebehandling 24022015
close all; clear all;
I = imread('pout.tif');
figure(1)
imshow(I,[])
impixelinfo
figure(2)
imhist(I)
ylim('auto')
% Histogramet viser at bildet har dårlig kontrast.
% Gråtoner i renge [75 -> 175]

% Strekker histogrammet med ein mappefunksjon
I1 = imadjust(I,[75/255 175/255],[0 1]);
figure(3)
imshow(I1,[])
figure(4)
imhist(I1)
ylim('auto')


% Strekking med klippeprosent, default klippeprosent er 
% 1% oppe og nede i histogrammet.
I3 = imadjust(I,stretchlim(I), [0 1]);
figure(5)
imshow(I3,[])
figure(6)
imhist(I3)
ylim('auto')


% Mappefunksjon med oppslagstabell (LUT- Look Up Taable)
LUT = uint8(zeros(1,256));
LUT (1:65) = 2*(0:64);
LUT (66:129) = 128;
LUT (130:256) = (130:256)-1;
figure(7)
plot(LUT)
title('Mappefunksjon')
xlabel('gråtoneverdi inn')
ylabel('Gråtoneverdi ut')
grid on

I4 = intlut(I,LUT);
figure(8)
imshow(I4,[])

% >>imtool (imadjust)

% Studerer imshow()
I5 = I*4;
figure(9)
imshow(I5,[])
% Byteklipping 255*4 = 1020 -> 255 pga uint8 tallområde.
I6=4*double(I);
figure(10)
imshow(16,[])

% Lager et støybilde
I7 = rand(256).*1000;
figure(11)
imshow(I7,[0 1000])
impixelinfo
I8 =I7-500;
figure(12)
imshow(I8,[-500 500])

%% Frekvensanalyse 27022015

close all; clear all;
%**********************
% 
% Eksperiment 1:
%xn = [5 2 -2 4];

%*********************
%
% Eksperiment 2:
%xn = [ 3 2 4 -1 2 1 4 -1];

%*********************
%
% Eksperiment 3:
% n= 0:1:92;
% xn = cos(n*pi/10);
%*********************
%
% Eksperiment 4:
n= 0:1:500;
xn = cos(n*pi/10) + 0.50*cos(n*pi/8);

%FFT- Fast Fourier Transform
N = length(xn);
Xk = fft(xn,N);
figure(1)
k = 0:1:N-1;
stem(k,abs(Xk),'r','linewidth',2);
title('N-punkts FFT')
ylabel('|X[k]|')
xlabel('komponentnummer (k)')
grid on
% Observerer at vi har like mange frekvenskomponenter som samples.


% DTFT- Diskrete Time Fourier Transformasjon

% X(omega)= sum(x[n]*exp(-j*n*omega))

omega = 0:0.001:2*pi;
X = 0;
for n = 0:1:N-1;
    X = X+xn(n+1)*exp(-1i*n*omega);
end
XX = abs(X);
figure(2)
plot(omega,XX); hold on
omega1 = k*2*pi/N;
stem(omega1,abs(Xk),'r','linewidth',2)
title('DTFT og DFT(FFT)')
xlabel('\Omega')
ylabel('|X(\Omega)|')
grid on

% Observerer at FFT er en sampla versjon av DTFT (som er kontinuerlig)
% Eksperiment 3 : Signal med en frenkvens. FFT og DTFT gir mange frekvens
%                 komponenter. Men vi får en kraftig "peak" der frekvensen
%                 til signalet er. 
%                 De andre komponentene er spektral- lekasje.
%% Frekvens analyse uten vindusfunksjon 03032015
close all; clear all;
f1 = 1/16; 
f2 = 3/8; 
fs = 6.8;
t = 0:1/fs:2000;
xn = cos(2*pi*f1*t)+0.50*cos(2*pi*f2*t);%+randn(size(t));
% Velger analysevindu (Tidsvindu)/ utsnitt av signalet.
%N = 2^10=1024
%FFT
N=1024;
figure(1)
plot(t,xn); hold on
plot(t(1:N),xn(1:N),'r'); hold off
% Tar nå N punkt FFT av utsnittet
Xk = fft(xn,N);
figure(2)
k = 0:1:length(Xk)-1;
stem(k,abs(Xk),'.r')
xlabel('komponent nummer (k)')
ylabel('|X[k]|')
title('N punkt FFT av xn = cos(2*pi*f1*t)+0.50*cos(2*pi*f2*t)')
grid on

% Reskalerer x-aksen til frekvens
f = (0:1/N:1/2)*fs;

figure(3)
Xk = abs(Xk);
stem(f,Xk(1:length(f)),'.r')
xlabel('Frekvens (Hz)')
ylabel('|X[k]|')
title('N punkt FFT av xn = cos(2*pi*f1*t)+0.50*cos(2*pi*f2*t)')
grid on

% Det er kun frekvensene fra 0-fs/2 som er intresante.
% Frekvenser fra fs/2 -fs kun symmetri(matematikk) "FAKE" 

% Vi zoomer inn på de frekvensene som er av intresse i linjespektret.

figure(4)
Xk = abs(Xk);
stem(f(1:100),Xk(1:100),'.r')
xlabel('Frekvens (Hz)')
ylabel('|X[k]|')
title('N punkt FFT av xn = cos(2*pi*f1*t)+0.50*cos(2*pi*f2*t)')
grid on

% Observerer to peaker ved to frekvenser f1 og f2
% Ser også at vi har spektral-lekasje.
% Dette kan reduseres med vindusfunksjon.

% Vi la på støy og reduserte amplituden på den ene cosinusen 
% I frekvens spekteret "drukna" komponenten i støygulvet.

figure(5)
Xk = abs(Xk);
plot(f(1:100),Xk(1:100),'r')
xlabel('Frekvens (Hz)')
ylabel('|X[k]|')
title('N punkt FFT av xn = cos(2*pi*f1*t)+0.50*cos(2*pi*f2*t)')
grid on

% Reskalerer y-aksen til dB. Dette for at vi skal få fram 
% både sterke og svake komponenter samtidig.

figure(6)
XKK = 20*log(Xk);
plot(f(1:100),XKK(1:100),'r')
xlabel('Frekvens (Hz)')
ylabel('|X[f]| i dB')
title('N punkt FFT av xn = cos(2*pi*f1*t)+0.50*cos(2*pi*f2*t)')
grid on

%% FFT nivå, effektspekter, periodogram 10032015 
close all; clear all;
f1 = 50;
f2 = 120;
f3 = 150;
fs = 1000;
t = 0:1/fs:16;
xn = sin(2*pi*f1*t)+2*sin(2*pi*f2*t)+0.25*sin(2*pi*f3*t);
nn = randn(size(xn));
% signal med støy
xx = xn+nn;

% FFT
N = 1024;
%N = 2048;
%N = 512;
xx1 = xx(1:N);
wnh = window(@hamming,N);
xnh = xx1.*wnh';
Xk = fft(xnh,N);
XK = 20*log10(Xk);
f = 0:(1/N)*fs:fs/2;
figure(1)
plot(f,XK(1:length(f)));
xlabel('Frekvens (Hz)')
ylabel('|X(f)| i dBV')
title('Linjespekter i dBV')
grid on

% Observerer at: Peak A=2V -> 54.6 dBV regna ut
%                             A = 1V -Zlog(1/2)=-6dB
%                             A = 0.25 -> 20log(2/0.25)=-18dB

% Periodogram (welch-metoden)
[Py,f] = pwelch(xx,hamming(N),N/4,N,fs);
PY = 10*log10(Py*fs);
figure(2)
plot(f,PY(1:length(f)));
xlabel('Frekvens (Hz)')
title('Effektspekter dBW')
grid on

% Observerer: Antall segment styrer spredninga på støygulvet.
%             Lenger logga data eller størrelse (gå ned på)
%             segment.
%             Men mindre segment fører til dårligere oppløsning 
%             delta f = (1/N)*fs 

% signal: N = 1024/hamming  A = 2V  A = 2X/N -> X = AN/2 = 1024
% 20*log1024 = 60dBv -5.4dB = 54,6 dBV

%% Effekt spekter og effekt tetthet 13032015

close all; clear all;
% Periodogram --> power spekter dBW, dBm.
%             --> power spekter density,(psd), dBW/Hz, dBm/hz


% Test signal fra førrige celle
f1 = 50;
f2 = 120;
f3 = 150;
fs = 1000;
t = 0:1/fs:16;
xn = sin(2*pi*f1*t)+2*sin(2*pi*f2*t)+0.25*sin(2*pi*f3*t);
nn = randn(size(xn));
N = 1024;
n= nn(1:N);
% signal med støy
xx = xn+nn;

N = 1024;
% PSD - Power spekter density (effekt tetthet i dBW/Hz)
% Ensidig (tosided) spekter
[Py1,f1]= pwelch(xx,hamming(N),N/4,N,fs,'twosided','psd');
PY1 = 10*log10(Py1);

figure(1)
plot(f1,PY1(1:length(f1)));
title('Effekt tetthet(psd) dBW/Hz')
xlabel('Frekvens (Hz)')
grid on
% Støy effekt 
Pn = (1/N)*sum(N.^2);

% Pn =1.0267 w ( støyeffekt)

PYY1 = 10*log10(Py1/0.001);

figure(2)
plot(f1,PYY1(1:length(f1)));
title('Effekt tetthet(psd) dBm/Hz, tosidig spekter')
xlabel('Frekvens (Hz)')
grid on


% PSD - Power spekter density (effekt tetthet i dBW/Hz)
% Ensidig (onesided) spekter
[Py2,f2]= pwelch(xx,hamming(N),N/4,N,fs,'onesided','psd');
PY2 = 10*log10(Py2/1);

figure(3)
plot(f2,PY2(1:length(f2)));
title('Effekt tetthet(psd) dBm/Hz, tosidig spekter')
xlabel('Frekvens (Hz)')
grid on

% PSD - Power spekter (effekt tetthet i dBW)
% tosidig spekter
[Py3,f3]= pwelch(xx,hamming(N),N/4,N,fs,'twosided','power');
PY3 = 10*log10(Py3/1);

figure(4)
plot(f3,PY3(1:length(f3)));
title('Effekt(psd) dBW, tosidig spekter')
xlabel('Frekvens (Hz)')
grid on

%% FIR Window 17.01.2015

close all; clear all;
% Ref. eksempel på tavla
% fs = 8kHz,fp = 1kHz, N = 11, rektangulært vindu (ikke vindu)

b11r = fir1(10,0.25,rectwin(11),'noscale');
b11rs = fir1(10,0.25,rectwin(11));
% skalere manuelt DC-gain = 1 (0dB)
bb=b11r/sum(b11r);
fvtool(b11r,1,b11rs,1,bb,1)

% Eksperiment 1:
% N = 11,51,101,151
fs = 8000;
fpass = 1000;
% normaliserer kantfrekvens
fp = fpass/(fs/2);
b11 = fir1(10,fp,rectwin(11));
b51 = fir1(50,fp,rectwin(51));
b101 = fir1(100,fp,rectwin(101));
b151 = fir1(150,fp,rectwin(151));
fvtool(b11,1,b51,1,b101,1,b151,1)
fvtool(b11,1,b51,1,b101,1,b151,1)


% Konklusjon: Steilheten (transisjons område(T.W.)) øker med 
%             antall koeffisienter. Rippel i pass og stopp båndet
%             er uforandra.

% Eksperiment 2:
% N = 101,Tester innvirkning ulike vindu.
b101r = fir1(100,fp,rectwin(101));
b101h = fir1(100,fp,window(@hamming,101));
b101han = fir1(100,fp,window(@hanning,101));
b101bh = fir1(100,fp,window(@blackmanharris,101));
b101ba = fir1(100,fp,window(@bartlett,101));
fvtool(b101r,1,b101h,1,b101han,1,b101bh,1,b101ba,1)
fvtool(b101r,1,b101h,1,b101han,1,b101bh,1,b101ba,1)


% Konklusjon: Dempinga regulerer vi med valg av vindu.
%             God demping går på bekostning av steilhet
%             Kantfrekvensene "flyter". Redesign -->



%% IIR Design 24.03.2015

close all; clear all;
% Analogt RC filter
% R= 1k, c = 1uf
% H(s) = 1000/(s+1000)
% Digital versjon (bilineær transformasjon)
% H(z)= (1/3)*(1+z^-1)/(1-(1/3)z^-1)
% fs = 1000
% Digital versjon:
fs =1000;
bd = [1/3 1/3];
ad = [1 -1/3];
[Hd,f] = freqz(bd,ad,1024,fs); hold on
figure(1)
plot(f,abs(Hd));
grid on
% Kommentar: Frekvensresponsen blir eksakt =0 ved fs/2.
%            Det er alltid slik med bilineær transformasjon.

% Analoge versjon (utgangspunktet)
ba = [1000];
aa= [1 1000];
Ha = freqs(ba,aa,2*pi*f);
plot(f,abs(Ha),'k'); hold off
legend('Digital','Analog')
xlabel('Frekvens (Hz)')
ylabel('|Hd(f)|,|Ha(f)|')
title('RC-filter analog/ digital versjon')
% Kantfrekvens:
%              -3dB 1/sqrt(2) --> fc(fpass)=1/(2*pi*R*C)
%                                 fc = 159Hz
% Konklusjon: Kantfrekvensen til det anloge filter (159Hz) blir noe
%             forvrengt (wraping).Digitalt  filter får kantfrekvens
%             ca.150Hz

% studerer fasen:
figure(2)
plot(f,angle(Hd)*180/pi); hold on
plot(f,angle(Ha)*180/pi,'k'); hold off
legend('Digital','Analog')
xlabel('Frekvens (Hz)')
ylabel('Fase')
title('RC-filter analog/ digital versjon')
% Konklusjon: fasen til det digitale filter er ulineær.
%             IIR filter er alltid slik. kan aaldri lage 
%             ifilter med lineær fase når IIR type.

%[z,p,k] = buttap(4);
%[b,a]=zp2tf(z,p,k)

%% IIR Filter Bilineær 27032015
close all; clear all;
% Digital spec.
fp = 1500;
fs = 8000;
% Butterworth N = 4
N = 4;
[b,a]= butter(N,2*pi*fp,'s');
% Frekvensresponsen analogt
[H,w] = freqs(b,a);
f = w/(2*pi);
figure(1)
plot(f,abs(H)); hold on
axis([0 4000 0 1.1])
grid on

% Lager digitalt filter 
% Bruker bilineær transformasjon
[b1, a1] = bilinear(b,a,fs);
% Frekvensrespons digitalt
[H1, w1] = freqs(b1,a1);
f1 = (w1/pi)*(fs/2);
plot(f1,abs(H1),'r');

% Konklusjon: kantfrekvensen er forvrengt (wraping)

% Pre- wraping av kantfrekvensen
fd = 1500;
fp1 = (fs/pi)*tan(pi*fd/fs);
% fp1 =1701
% Nytt analogt filter 
[b2,a2]= butter(N,2*pi*fp1,'s');
% Frekvens responsen prewrapa analogt filter 
[H2,w2] = freqs(b2,a2);
f2 =w2/(2*pi);
plot(f2,abs(H2),'k');

% Lager nytt digitalt filter
[b3,a3]= bilinear(b2,a2,fs);
% Frekvensresponsen
[H3, w3] = freqs(b3,a3);
f3 = (w3/pi)*(fs/2);
plot(f3,abs(H3),'g'); hold off
legend('Analogt filt','Digitalt filt forvrengt',...
        'Analogt filt for- forvrengt', 'Digitalt filt korrigert')
xlabel('frekvens (Hz)')
title('Bilineær transformasjon, Butterworth N = 4')
ylabel('Frekvensresponsen |H(f)| Lin skala')

% ********************************************
% Start IIR filter design fra spesifikasjon
fpass = 1500;
fstopp = 2500;
fs = 8000;
Rp = 0.10; % 0.1 dB Rippel i passbåndet
Rs = 80;   % 80 dB demping i stoppbåndet
[N,wn]= buttord(fpass/(fs/2),fstopp/(fs/2),Rp,Rs,'s');

% N = 22


%% Korrolasjon 10.04.2015

close all; clear all;
% Støy signal
xn = 0.1*randn(1,100);
yn = [0 0 0 0 xn(1:96)];

% Eksperiment 1:
% Krysskorrolasjon mellom xn og yn (forsinka versjon av xn)
[rxy,lag]= xcorr(xn,yn);
figure(1)
subplot(3,1,1)
stem(xn,'.r')
title('Støy x[n]')
subplot(3,1,2)
stem(yn,'.r')
title('Forsinka Støy x[n]= xx[n-4]')
subplot(3,1,3)
stem(lag,rxy,'.');
title('Krysskorrelasjon rxy(lag)')
xlabel('Lag')

% Konklusjon: Krysskorrolasjon (max) avdekker forskyvninga.

% Eksperiment 2:
% Autokorrolasjon (ACF)
[rxx,lag]=xcorr(xn,xn,'biased');
figure(2)
subplot(3,1,1)
stem(xn,'.r')
title('Støy x[n]')
subplot(3,1,2)
stem(lag,rxx,'.');
title('Autokorrelasjon (ACF) rxx(lag)')
xlabel('Lag')
% Beregner effekten i støysignalet
Pn=(1/length(xn))*sum(xn.^2);
% Konklusjon:
% rxx(0) (ACF(0)) = effekten i signalet Pn = 0.01w
% Ellers er rxx(lag)=0

% Eksperiment 3:
% Krysskorrolasjon av et signal med støy og kjent sekvens 
yy=[1 1 -1 -1 1 -1]; % Kjent sekvens
xx=[zeros(1,50) yy zeros(1,44)];
yn = xx+xn;
[ry_yy,lag]=xcorr(yn,yy);
figure(3)
subplot(3,1,1)
stem(yn,'.')
title('Sekvens med støy')
subplot(3,1,2)
stem(lag,ry_yy,'.')
title('Krysskorrelasjon mellom sekvens med støy og kjent sekvens')


% Eksperiment 4:
% Korrelasjon mellom sin() og cos()
n = 0:1:99;
%xn = cos((pi/10)*n);
xn = cos((pi/10)*n)+cos((pi/20)*n);
%yn = sin(pi/10);
yn = sin((pi/10)*n);
[rxx,lag1]= xcorr(xn,xn,'unbiased');
[ryy,lag2]= xcorr(yn,yn,'unbiased');
[rxy,lag3]= xcorr(xn,yn,'unbiased');
figure(4)
subplot(3,1,1)
stem(xn,'.b');hold on
plot(yn,'k'); hold off
subplot(3,1,2)
stem(lag1,rxx,'.b');hold on
plot(lag2,ryy,'k'); hold off
title('Autokorrelasjon')
xlabel('lag')
% Konklusjon:
%            1. sin() og cos() har begge cos() som ACF
%            2. ACF avdekker perioden til signalet.
%            3. ACF(0)= Ps (effekten i signalet)


subplot(3,1,3)
stem(lag3,rxy,'.r');
title('Krysskorrelasjon rxy(lag)')
xlabel('Lag')

%% Nedsampling/ oppsampling 14.04.2015
close all; clear all;
fs = 1000;
f1 = 50; 
f2 = 100;
A1 = 1;
A2 = 0.50;
t = 0:1/fs:10;
xn = A1*sin(2*pi*f1*t)+A2*sin(2*pi*f2*t);


% Eksperiment 1:
%               Nedsampling uten pre- desimasjonsfilter
% Nedsamplingsfaktor = M
M = 4;
yn1 = downsample(xn,M);
n = 1:M:100;
figure(1)
subplot(2,1,1)
stem(xn(1:100),'.'); hold on
stem(n,yn1(1:length(n)),'.r'); hold off
title('Orginal sekvens')
xlabel('sample nummer (n)')
subplot(2,1,2)
stem(n,yn1(1:length(n)),'.r');
title('Nedsampling uten pre- desimasjonsfilter')

% Konklusjon: Går bare dersom M =4

% Eksperiment 2:
%               Nedsampling med pre-desimasjons filter.

yn2 = decimate(xn,M);

figure(2)
subplot(2,1,1)
stem(xn(1:100),'.'); hold on
stem(n,yn2(1:length(n)),'.r'); hold off
title('Orginal sekvens')
xlabel('sample nummer (n)')
subplot(2,1,2)
stem(n,yn2(1:length(n)),'.r');
title('Nedsampling med pre- desimasjonsfilter')

% >> fdatool design multirate filter.

% eksperiment 3:
%               Oppsampling med interpulasjonsfilter
% Oppsamplingsfaktor =1
L = 4;
n1 = 1:L:400;
yn3 = interp(xn,L);
figure(3)
subplot(2,1,1)
stem(yn3(1:400),'.'); hold on
stem(n1,xn(1:100),'.r'); hold off
title('Orginal sekvens')
subplot(2,1,2)
stem(n1,yn3(1:100),'.r'); hold off
title('Oppsampling med interpolasjonsfilter')


% Eksperiment 4:
%               Oppsampling uten interpolasjonsfilter
yn4 = upsample(xn,L);
figure(4)
subplot(2,1,1)
stem(yn4(1:100),'.');


% Eksperiment 5:
%               Ny samplefrekvens fs' = (2/3)*fs
yn5 = resample(xn,2,3);
subplot(2,1,2)