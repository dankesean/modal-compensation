%% Modal error analysis and compensation code - Koopman for compensation
% Author - Sean McGowan, University of Adelaide
% Date - 6/5/24

clear all
close all
clc

%% Load data and split into training and testing

load('gulfstream_data.mat')

Tspan_train = 1:(9*366+27*365);
Tspan_test = 1:(9*366+28*365); 

SST_sat = SST_prediction_sat(:,:,Tspan_train);
SST_model = SST_prediction_model(:,:,Tspan_train);

%% Plot satellite and model data

Tlim_full = [floor(min(SST_sat,[],'all')) ceil(max(SST_sat,[],'all'))];

figure
subplot(1,2,1)
h1 = imagesc(lat_sat(indexX(1):indexX(2)),lon_sat(indexY(1):indexY(2)),zeros(50,50));
xlim(lat_sat([indexX(1) indexX(2)-1]))
ylim(lon_sat([indexY(1) indexY(2)-1]))
set(gca,'Xdir','reverse','Ydir','reverse')
view(270,270)
colormap turbo
clim(Tlim_full)
title('Satellite data')

subplot(1,2,2)
h2 = imagesc(lat_sat(indexX(1):indexX(2)),lon_sat(indexY(1):indexY(2)),zeros(50,50));
xlim(lat_sat([indexX(1) indexX(2)-1]))
ylim(lon_sat([indexY(1) indexY(2)-1]))
set(gca,'Xdir','reverse','Ydir','reverse')
view(270,270)
colormap turbo
clim(Tlim_full)
title('Model data')
    
for i=length(Tspan_train)-365:length(Tspan_train)
    set(h1,'cdata',SST_sat(:,:,i)) 
    set(h2,'cdata',SST_model(:,:,i)) 
    
    sgtitle(datestr(datetime(1985,1,1)+i-1))
    
    drawnow;
    pause(0.01);
end

%% Plot discrepancy

SST_disc = SST_sat-SST_model;

Tlim_disc = [floor(min(SST_disc,[],'all')) ceil(max(SST_disc,[],'all'))];

figure
h1 = imagesc(lat_sat(indexX(1):indexX(2)),lon_sat(indexY(1):indexY(2)),zeros(50,50));
xlim(lat_sat([indexX(1) indexX(2)-1]))
ylim(lon_sat([indexY(1) indexY(2)-1]))
set(gca,'Xdir','reverse','Ydir','reverse')
view(270,270)
colorbar
colormap turbo
clim(Tlim_disc)
    
for i=length(Tspan_train)-365:length(Tspan_train)
    set(h1,'cdata',SST_disc(:,:,i)) 
    
    sgtitle(datestr(datetime(1985,1,1)+i-1))
    
    drawnow;
    pause(0.01);
end

%% Select delayed signals
% in this example single-step delays are used however, nonuniform delays
% may be investigated by adapting the Delay_vec vector

Ndelays = 10;
Delay_vec = ones(1,Ndelays);
Delay_vec = [0 Delay_vec];


%% DMD with delay

dt = 1;
window = floor(length(Tspan_train)-sum(Delay_vec)-1);
dims = size(permute(SST_disc,[3 1 2]));

% create Hankel snapshot matrix
Snapshots_hankel = zeros(prod(dims(2:end))*(Ndelays+1), window+1);
for i = 1:(window+1)
    for j = 1:(Ndelays+1)
        Snapshots_hankel(1+(j-1)*prod(dims(2:end)):j*prod(dims(2:end)),i) = ...
            reshape(SST_disc(:,:,length(Tspan_train)-sum(Delay_vec(1:j))-(window+1)+i),1,prod(dims(2:end)));
    end
end

Snapshots_hankel(isnan(Snapshots_hankel))=0;

% create snapshot matrices
H = Snapshots_hankel(:,1:end-1);
Hprime = Snapshots_hankel(:,2:end);

% for reduced order 
[U, S, V] = svds(H,5000);

% calculate reduced order approximation of linear matrix A
A_tilde_H = U'*Hprime*V/S;

clear U Snapshots_hankel

[Eigenvectors_H, Eigenvalues_H] = eig(A_tilde_H);
Eigenvectors_H = Hprime*V*inv(S)*Eigenvectors_H;
Eigenvalues_H = diag(Eigenvalues_H);

clear V S

% find mode amplitudes
ModeAmplitudes_H = Eigenvectors_H\H(:,1);

% find mode frequencies

ModeFrequencies_H = (angle(Eigenvalues_H)/pi)/(2*dt);

% calculate growth rates from eigenvalues
GrowthRates_H = log(Eigenvalues_H)/dt;

Eigenvectors_H(Eigenvectors_H==0) = nan;


%% Plot eigenvalues

figure
plot(exp(1i*linspace(-pi,pi,500)),'b--')
hold on
plot(Eigenvalues_H(Eigenvalues_H.*conj(Eigenvalues_H)<=1),'k.')
plot(Eigenvalues_H(Eigenvalues_H.*conj(Eigenvalues_H)>1),'r.')
title('DMD with delay eigenvalues')

%% Plot dominant modes 

% order modes by polar angle, or modulus
[~, Mode_Index] = sort(abs(angle(Eigenvalues_H)),'ascend');
%[~, Mode_Index] = sort(abs(Eigenvalues_H),'descend');

ModeAmplitudes_H_prediction = ModeAmplitudes_H_prediction(Mode_Index);
ModeAmplitudes_H = ModeAmplitudes_H(Mode_Index);
GrowthRates_H = GrowthRates_H(Mode_Index);
Eigenvectors_H = Eigenvectors_H(:,Mode_Index);
Eigenvalues_H = Eigenvalues_H(Mode_Index);

Modes_plot = 1:2:15;

figure
for i = 1:8
    subplot(2,4,i)
    imagesc(lat_sat(indexX(1):indexX(2)),lon_sat(indexY(1):indexY(2)),reshape(...
        real(Eigenvectors_H(1:prod(dims(2:end)),Modes_plot(i))),dims(2:3)))
    xlim(lat_sat([indexX(1) indexX(2)-1]))
    ylim(lon_sat([indexY(1) indexY(2)-1]))
    set(gca,'Xdir','reverse','Ydir','reverse')
    view(270,270)
    grid off
    colormap turbo
end


%% DMD with delay reconstruction 

r_reduced = size(Eigenvectors_H,2);

figure
subplot(1,2,1);
h1 = imagesc(lat_sat(indexX(1):indexX(2)),lon_sat(indexY(1):indexY(2)),zeros(50,50));
xlim(lat_sat([indexX(1) indexX(2)-1]))
ylim(lon_sat([indexY(1) indexY(2)-1]))
set(gca,'Xdir','reverse','Ydir','reverse')
view(270,270)
title('True')
colorbar
colormap turbo
clim(Tlim_disc)

subplot(1,2,2)
h2 = imagesc(lat_sat(indexX(1):indexX(2)),lon_sat(indexY(1):indexY(2)),zeros(50,50));
xlim(lat_sat([indexX(1) indexX(2)-1]))
ylim(lon_sat([indexY(1) indexY(2)-1]))
set(gca,'Xdir','reverse','Ydir','reverse')
view(270,270)
title('DMD with delay reconstruction')
colorbar
colormap turbo
clim(Tlim_disc)

for i = length(Tspan_train)-365:length(Tspan_train)
    % Hankel DMD reconstruction
    SST_recon_H = real(Eigenvectors_H(1:prod(dims(2:end)),...
        1:r_reduced)*(exp(GrowthRates_H(1:r_reduced).*Tspan_train(i+1-sum(Delay_vec))).*...
        ModeAmplitudes_H(1:r_reduced)));
    
    % reshape back to original dimensions
    SST_recon_H = reshape(SST_recon_H,dims(2:3));

    set(h2,'cdata',SST_recon_H)
    set(h1,'cdata',SST_disc(:,:,Tspan_train(i+3)))
    
    drawnow;
    pause(0.01);
end

%% Mode amplitudes for prediction

Eigenvectors_H(isnan(Eigenvectors_H)) = 0;
ModeAmplitudes_H_prediction = Eigenvectors_H\H(:,end);
Eigenvectors_H(Eigenvectors_H==0) = nan;

% order modes by mode amplitude, polar angle, or modulus
%[~, Mode_Index] = sort(ModeAmplitudes_H_prediction.*conj(ModeAmplitudes_H_prediction),'descend');
[~, Mode_Index] = sort(abs(angle(Eigenvalues_H)),'ascend');
%[~, Mode_Index] = sort(abs(Eigenvalues_H),'descend');

ModeAmplitudes_H_prediction = ModeAmplitudes_H_prediction(Mode_Index);
ModeAmplitudes_H = ModeAmplitudes_H(Mode_Index);
GrowthRates_H = GrowthRates_H(Mode_Index);
Eigenvectors_H = Eigenvectors_H(:,Mode_Index);
Eigenvalues_H = Eigenvalues_H(Mode_Index);

%% Prediction compensation

figure
subplot(1,4,1);
h1 = imagesc(lat_sat(indexX(1):indexX(2)),lon_sat(indexY(1):indexY(2)),zeros(50,50));
xlim(lat_sat([indexX(1) indexX(2)-1]))
ylim(lon_sat([indexY(1) indexY(2)-1]))
set(gca,'Xdir','reverse','Ydir','reverse')
view(270,270)
grid off
title('True')
colormap turbo
clim(Tlim_full)

subplot(1,4,2);
h2 = imagesc(lat_sat(indexX(1):indexX(2)),lon_sat(indexY(1):indexY(2)),zeros(50,50));
xlim(lat_sat([indexX(1) indexX(2)-1]))
ylim(lon_sat([indexY(1) indexY(2)-1]))
set(gca,'Xdir','reverse','Ydir','reverse')
view(270,270)
grid off
title('Model')
colormap turbo
clim(Tlim_full)

subplot(1,4,3)
h3 = imagesc(lat_sat(indexX(1):indexX(2)),lon_sat(indexY(1):indexY(2)),zeros(50,50));
xlim(lat_sat([indexX(1) indexX(2)-1]))
ylim(lon_sat([indexY(1) indexY(2)-1]))
set(gca,'Xdir','reverse','Ydir','reverse')
view(270,270)
grid off
title('Compensated model')
colormap turbo
clim(Tlim_full)

h = colorbar;
set(h,'position',[0.8,0.1,0.02,0.75])
h.Label.String = 'T (K)';
h.Label.FontSize = 12;

predict_length = 90;

E_model = [];
E_H = [];
E_spatial_model = zeros(50,50,predict_length+1);
E_spatial_H = zeros(50,50,predict_length+1);
% prediction for half a year
for i = length(Tspan_train):length(Tspan_train)+predict_length
    % Hankel DMD reconstruction
    SST_recon_H = real(Eigenvectors_H(1:prod(dims(2:end)),...
        1:r_reduced)*(exp(GrowthRates_H(1:r_reduced).*...
        (Tspan_test(i+1-sum(Delay_vec))-Tspan_train(end-sum(Delay_vec)))).*...
        ModeAmplitudes_H_prediction(1:r_reduced)));
    
    % reshape back to original dimensions
    SST_recon_H = reshape(SST_recon_H,dims(2:3))+SST_prediction_model(:,:,i);
    
    set(h3,'cdata',SST_recon_H)
    set(h2,'cdata',SST_prediction_model(:,:,i))
    set(h1,'cdata',SST_prediction_sat(:,:,i))
    %sgtitle(datestr(datetime(2014,1,1)+i-1))
    
    E_model = [E_model (1/prod(dims(2:end))*sum((SST_prediction_model(:,:,i)-SST_prediction_sat(:,:,i)).^2,'all'))^0.5];
    E_H = [E_H (1/prod(dims(2:end))*sum((SST_recon_H-SST_prediction_sat(:,:,i)).^2,'all'))^0.5];
    
    E_spatial_model(:,:,i-length(Tspan_train)+1) = (SST_prediction_model(:,:,i)-SST_prediction_sat(:,:,i)).^2;
    E_spatial_H(:,:,i-length(Tspan_train)+1) = (SST_recon_H-SST_prediction_sat(:,:,i)).^2;
    
    drawnow;
    pause(0.05);
end

figure
plot(E_model,'color',[0.1855 0.4989 0.2322])
hold on
plot(E_H,'b')
ylabel('Spatially averaged error','interpreter','latex')
xlabel('Time (days)','interpreter','latex')
xlim([0 90])

RMSE_model = sqrt((1./(1:length(E_model))).*cumsum(E_model.^2));
RMSE_H = sqrt((1./(1:length(E_H))).*cumsum(E_H.^2));

figure
plot(RMSE_model,'color',[0.1855 0.4989 0.2322])
hold on
plot(RMSE_H,'b')
ylabel('RMSE(t)','interpreter','latex')
xlabel('Time (days)','interpreter','latex')
xlim([0 90])

E_spatial_model_avg = sqrt(1/(predict_length+1)*sum(E_spatial_model,3));
E_spatial_H_avg = sqrt(1/(predict_length+1)*sum(E_spatial_H,3));
 
figure
subplot(1,3,1)
imagesc(lat_sat(indexX(1):indexX(2)),lon_sat(indexY(1):indexY(2)),E_spatial_model_avg)
title('Model')
xlim(lat_sat([indexX(1) indexX(2)-1]))
ylim(lon_sat([indexY(1) indexY(2)-1]))
clim([min([E_spatial_model_avg E_spatial_H_avg],[],'all'),max([E_spatial_model_avg E_spatial_H_avg],[],'all')])
colormap turbo

set(gca,'Xdir','reverse','Ydir','reverse')
view(270,270)

subplot(1,3,2)
imagesc(lat_sat(indexX(1):indexX(2)),lon_sat(indexY(1):indexY(2)),E_spatial_H_avg)
title('DMD compensated model')
xlim(lat_sat([indexX(1) indexX(2)-1]))
ylim(lon_sat([indexY(1) indexY(2)-1]))
clim([min([E_spatial_model_avg E_spatial_H_avg],[],'all'),max([E_spatial_model_avg E_spatial_H_avg],[],'all')])
colormap turbo

set(gca,'Xdir','reverse','Ydir','reverse')
view(270,270)

h = colorbar;
set(h,'position',[0.75,0.1,0.02,0.75])
h.Label.String = 'E';
h.Label.FontSize = 12;

