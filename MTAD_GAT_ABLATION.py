from lib import *
import os
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import average_precision_score

class CustomMTADGAT(nn.Module):
    """ MTAD-GAT model class.

    :param n_features: Number of input features
    :param window_size: Length of the input sequence
    :param out_dim: Number of features to output
    :param kernel_size: size of kernel to use in the 1-D convolution
    :param feat_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in feat-oriented GAT layer
    :param time_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in time-oriented GAT layer
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param gru_n_layers: number of layers in the GRU layer
    :param gru_hid_dim: hidden dimension in the GRU layer
    :param forecast_n_layers: number of layers in the FC-based Forecasting Model
    :param forecast_hid_dim: hidden dimension in the FC-based Forecasting Model
    :param recon_n_layers: number of layers in the GRU-based Reconstruction Model
    :param recon_hid_dim: hidden dimension in the GRU-based Reconstruction Model
    :param dropout: dropout rate
    :param alpha: negative slope used in the leaky rely activation function

    """

    def __init__(
        self,
        n_features,
        window_size,
        out_dim,
        kernel_size=7,
        feat_gat_embed_dim=None,
        time_gat_embed_dim=None,
        use_gatv2=True,
        gru_n_layers=1,
        gru_hid_dim=150,
        forecast_n_layers=1,
        forecast_hid_dim=150,
        recon_n_layers=1,
        recon_hid_dim=150,
        dropout=0.2,
        alpha=0.2,
        nets = ['feature_gat', 'temporal_gat', 'forecasting_model', 'recon_model']
    ):
        super(CustomMTADGAT, self).__init__()
        
        self.conv = ConvLayer(n_features, kernel_size)
        
        self.feature_g = False
        self.temporal_g = False
        
        if 'feature_gat' in nets:
            self.feature_g = True
            self.feature_gat = FeatureAttentionLayer(n_features, window_size, dropout, alpha, feat_gat_embed_dim, use_gatv2)
        
        if 'temporal_gat' in nets:
            self.temporal_g = True
            self.temporal_gat = TemporalAttentionLayer(n_features, window_size, dropout, alpha, time_gat_embed_dim, use_gatv2)
        
        if 'feature_gat' in nets and 'temporal_gat' in nets:
            self.gru = GRULayer(3 * n_features, gru_hid_dim, gru_n_layers, dropout)
        
        elif 'feature_gat' in nets or 'temporal_gat' in nets:
            self.gru = GRULayer(2 * n_features, gru_hid_dim, gru_n_layers, dropout)
        
        else:
            self.gru = GRULayer(n_features, gru_hid_dim, gru_n_layers, dropout)

        self.forecasting = False
        self.recon = False

        if 'forecasting_model' in nets:
            self.forecasting = True
            self.forecasting_model = Forecasting_Model(gru_hid_dim, forecast_hid_dim, out_dim, forecast_n_layers, dropout)
        
        if 'recon_model' in nets:
            self.recon = True
            self.recon_model = ReconstructionModel(window_size, gru_hid_dim, recon_hid_dim, out_dim, recon_n_layers, dropout)

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features

        x = self.conv(x)
        
        if self.feature_g:
            h_feat = self.feature_gat(x)
        
        if self.temporal_g:
            h_temp = self.temporal_gat(x)

        if self.feature_g and self.temporal_g:
            h_cat = torch.cat([x, h_feat, h_temp], dim=2)  # (b, n, 3k)
        elif self.feature_g:
            h_cat = torch.cat([x, h_feat], dim=2)  # (b, n, 2k)
        elif self.temporal_g:
            h_cat = torch.cat([x, h_temp], dim=2)  # (b, n, 2k)
        else:
            h_cat = x

        _, h_end = self.gru(h_cat)
        
        h_end = h_end.view(x.shape[0], -1)   # Hidden state for last timestamp

        if self.forecasting:
            predictions = self.forecasting_model(h_end)
        if self.recon:
            recons = self.recon_model(h_end)

        if self.forecasting and self.recon:
            return predictions, recons
        if self.forecasting:
            return predictions
        if self.recon:
            return recons
            

def AblationTest(nets, x_train_path, x_test_path, y_test_path):
    All_AUROC = []
    All_AUPR = []

    PREDS_AUROC = []
    PREDS_AUPR = []

    RECONSTRUCT_AUROC = []
    RECONSTRUCT_AUPR = []

    for ii in range(len(x_train_path)):

        f = open(x_train_path[ii], "rb")
        x_train = pickle.load(f)
        f.close()

        f = open(x_test_path[ii], "rb")
        x_test = pickle.load(f)
        f.close()

        f = open(y_test_path[ii], "rb")
        y_test = pickle.load(f).reshape((-1))
        f.close()

        print('-------------------------------------------------------------')
        print(x_train_path[ii].split('/')[-1][:-4])

        x_train, scaler = normalize_data(x_train, scaler=None)
        x_test, _ = normalize_data(x_test, scaler=scaler)

        n_features = x_train.shape[1]
        window_size, target_dims = 100, x_train.shape[1]
        out_dim = 1
        batch_size, val_split, shuffle_dataset = 128, 0.2, True

        train_dataset = SlidingWindowDataset(x_train, window_size, target_dims)
        test_dataset = SlidingWindowDataset(x_test, window_size, target_dims)

        train_loader, val_loader, test_loader = create_data_loaders(
            train_dataset, batch_size, val_split, shuffle_dataset, test_dataset=test_dataset
        )

        device = torch.device("cuda:0")
        
        model = CustomMTADGAT(
            n_features,
            window_size,
            n_features,
            kernel_size=7,
            use_gatv2=True,
            feat_gat_embed_dim=None,
            time_gat_embed_dim=None,
            gru_n_layers=1,
            gru_hid_dim=300,
            recon_n_layers=1,
            recon_hid_dim=300,
            dropout=0.3,
            alpha=0.2,
            nets = nets
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr= 0.001)
        forecast_criterion = nn.MSELoss()
        recon_criterion = nn.MSELoss()
        epochs = 100

        save_path = 'Model/' + x_train_path[ii].split('/')[-1][:-4]+'_default.p'

        min_loss = 999999

        for e in tqdm_notebook(range(epochs)):
            model.train()

            forecast_b_losses = []
            recon_b_losses = []

            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()

                # reconstruction module과 forecasting module이 모두 있는 경우
                if ('recon_model' in nets) and ('forecasting_model' in nets):
                    preds, recons = model(x)
                    recon_loss = torch.sqrt(recon_criterion(x, recons))
                    forecast_loss = torch.sqrt(forecast_criterion(y, preds))
                    
                    all_loss = recon_loss + forecast_loss
                    all_loss.backward()
                    optimizer.step()

                # reconstruction module만 있는 경우
                elif ('recon_model' in nets):
                    recons = model(x)

                    all_loss = torch.sqrt(recon_criterion(x, recons))
                    all_loss.backward()
                    optimizer.step()

                # forecasting module만 있는 경우
                elif ('forecasting_model' in nets):
                    preds = model(x)

                    all_loss = torch.sqrt(forecast_criterion(y, preds))
                    all_loss.backward()
                    optimizer.step()


            model.eval()
            valid_losses = []

            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                # reconstruction module과 forecasting module이 모두 있는 경우
                if ('recon_model' in nets) and ('forecasting_model' in nets):
                    preds, recons = model(x)
                    recon_loss = torch.sqrt(recon_criterion(x, recons))
                    forecast_loss = torch.sqrt(forecast_criterion(y, preds))
                    valid_losses.append(recon_loss.item() + forecast_loss.item())

                # reconstruction module만 있는 경우
                elif ('recon_model' in nets):
                    recons = model(x)
                    recon_loss = torch.sqrt(recon_criterion(x, recons))
                    valid_losses.append(recon_loss.item())

                # forecasting module만 있는 경우
                elif ('forecasting_model' in nets):
                    preds = model(x)
                    forecast_loss = torch.sqrt(forecast_criterion(y, preds))
                    valid_losses.append(forecast_loss.item())

            valid_losses = np.array(valid_losses) 
            valid_losses = np.sqrt((valid_losses ** 2).mean())

            if min_loss > valid_losses:
                min_loss = valid_losses
                torch.save(model.state_dict(), save_path)

        model.load_state_dict(torch.load(save_path))

        if ('recon_model' in nets) and ('forecasting_model' in nets):
            preds = []
            recons = []
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(device)
                    y = y.to(device)

                    y_hat, _ = model(x)

                    # Shifting input to include the observed value (y) when doing the reconstruction
                    recon_x = torch.cat((x[:, 1:, :], y), dim=1)
                    _, window_recon = model(recon_x)

                    preds.append(y_hat.detach().cpu().numpy())
                    # Extract last reconstruction only
                    recons.append(window_recon[:, -1, :].detach().cpu().numpy())

            preds = np.concatenate(preds, axis=0)
            recons = np.concatenate(recons, axis=0)

            scaler = MinMaxScaler()

            preds = scaler.fit_transform((preds - x_test[100:])**2)
            recons = scaler.fit_transform((recons - x_test[100:])**2)

            print('Sum Performance')
            print(f'AUROC: {roc_auc_score(y_test[100:], preds.sum(1) + recons.sum(1))}')
            print(f'AUPR: {average_precision_score(y_test[100:], preds.sum(1) + recons.sum(1))}')

            All_AUROC.append(roc_auc_score(y_test[100:], preds.sum(1) + recons.sum(1)))
            All_AUPR.append(average_precision_score(y_test[100:], preds.sum(1) + recons.sum(1)))

            print('Predict Performance')
            print(f'AUROC: {roc_auc_score(y_test[100:], preds.sum(1))}')
            print(f'AUPR: {average_precision_score(y_test[100:], preds.sum(1))}')

            PREDS_AUROC.append(roc_auc_score(y_test[100:], preds.sum(1)))
            PREDS_AUPR.append(average_precision_score(y_test[100:], preds.sum(1)))

            print('Reconstruct Performance')
            print(f'AUROC: {roc_auc_score(y_test[100:], recons.sum(1))}')
            print(f'AUPR: {average_precision_score(y_test[100:], recons.sum(1))}')

            RECONSTRUCT_AUROC.append(roc_auc_score(y_test[100:], recons.sum(1)))
            RECONSTRUCT_AUPR.append(average_precision_score(y_test[100:], recons.sum(1)))
        elif ('recon_model' in nets):
            recons = []
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(device)
                    y = y.to(device)
                    recon_x = torch.cat((x[:, 1:, :], y), dim=1)
                    window_recon = model(recon_x)

                    recons.append(window_recon[:, -1, :].detach().cpu().numpy())

            recons = np.concatenate(recons, axis=0)

            scaler = MinMaxScaler()

            recons = scaler.fit_transform((recons - x_test[100:])**2)

            print('Reconstruct Performance')
            print(f'AUROC: {roc_auc_score(y_test[100:], recons.sum(1))}')
            print(f'AUPR: {average_precision_score(y_test[100:], recons.sum(1))}')

            RECONSTRUCT_AUROC.append(roc_auc_score(y_test[100:], recons.sum(1)))
            RECONSTRUCT_AUPR.append(average_precision_score(y_test[100:], recons.sum(1)))

        elif ('forecasting_model' in nets):
            preds = []
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(device)
                    y = y.to(device)

                    y_hat = model(x)

                    preds.append(y_hat.detach().cpu().numpy())

            preds = np.concatenate(preds, axis=0)

            scaler = MinMaxScaler()

            preds = scaler.fit_transform((preds - x_test[100:])**2)

            print('Predict Performance')
            print(f'AUROC: {roc_auc_score(y_test[100:], preds.sum(1))}')
            print(f'AUPR: {average_precision_score(y_test[100:], preds.sum(1))}')

            PREDS_AUROC.append(roc_auc_score(y_test[100:], preds.sum(1)))
            PREDS_AUPR.append(average_precision_score(y_test[100:], preds.sum(1)))

    if ('recon_model' in nets) and ('forecasting_model' in nets): 
        print(f'Predict and Reconstruction based Performance (AUROC, AUPR): {np.mean(All_AUROC), np.mean(All_AUPR)}')
        print(f'Predict based Performance (AUROC, AUPR): {np.mean(PREDS_AUROC), np.mean(PREDS_AUPR)}')
        print(f'Reconstruction based Performance (AUROC, AUPR): {np.mean(RECONSTRUCT_AUROC), np.mean(RECONSTRUCT_AUPR)}')

    elif ('recon_model' in nets):
        print(f'Reconstruction based Performance (AUROC, AUPR): {np.mean(RECONSTRUCT_AUROC), np.mean(RECONSTRUCT_AUPR)}')

        
    elif ('forecasting_model' in nets):
        print(f'Predict based Performance (AUROC, AUPR): {np.mean(PREDS_AUROC), np.mean(PREDS_AUPR)}')