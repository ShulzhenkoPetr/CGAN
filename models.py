import torch
from torch import nn


class FCBlock(nn.Module):

    def __init__(self, 
                 in_dim: int, 
                 out_dim: int, 
                 dropout_rate: float = 0.2, 
                 activation_type: str = 'relu'):
        super().__init__()

        self.fc = nn.Linear(in_dim, out_dim)
        if activation_type == 'tanh':
            self.activation = nn.Tanh() 
        elif activation_type == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate != 0 else nn.Identity()

    def forward(self, x):
        x = self.fc(x)
        x = self.dropout(x)
        return self.activation(x)


class FCGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.channels, self.img_size = config.img_channels, config.img_size
        self.do_rate = config.dropout_rate

        self.z_part = nn.ModuleList(
            [FCBlock(
                config.gen_z_fcs[i], 
                config.gen_z_fcs[i + 1], 
                dropout_rate=self.do_rate, 
                activation_type='relu') 
            for i in range(len(config.gen_z_fcs) - 1)]
        )
        self.y_part = nn.ModuleList(
            [FCBlock(
                config.gen_y_fcs[i], 
                config.gen_y_fcs[i + 1],
                dropout_rate=self.do_rate, 
                activation_type='relu') 
                for i in range(len(config.gen_y_fcs) - 1)]
        )
        self.joint_part = nn.ModuleList(
            [FCBlock(
                config.gen_j_fcs[i], 
                config.gen_j_fcs[i + 1],
                dropout_rate=self.do_rate, 
                activation_type='relu') 
                for i in range(len(config.gen_j_fcs) - 2)]
        )
        self.joint_part.append(FCBlock(
            config.gen_j_fcs[len(config.gen_j_fcs) - 2], 
            config.gen_j_fcs[len(config.gen_j_fcs) - 1],
            dropout_rate=self.do_rate, 
            activation_type='tanh'
        ))

    def forward(self, z, y):
        for z_emb_layer in self.z_part:
            z = z_emb_layer(z)

        for y_emb_layer in self.y_part:
            y = y_emb_layer(y)

        j_emb = torch.cat((z, y), dim=1)
        
        for upscale_layer in self.joint_part:
            j_emb = upscale_layer(j_emb)

        return j_emb.view(-1, self.channels, self.img_size, self.img_size)


class FCDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.do_rate = config.dropout_rate

        self.img_part = nn.ModuleList(
            [FCBlock(
                config.dis_z_fcs[i], 
                config.dis_z_fcs[i + 1], 
                dropout_rate=self.do_rate, 
                activation_type='relu') 
            for i in range(len(config.dis_z_fcs) - 1)]
        )
        self.y_part = nn.ModuleList(
            [FCBlock(
                config.dis_y_fcs[i], 
                config.dis_y_fcs[i + 1],
                dropout_rate=self.do_rate, 
                activation_type='relu') 
                for i in range(len(config.dis_y_fcs) - 1)]
        )
        self.joint_part = nn.ModuleList(
            [FCBlock(
                config.dis_j_fcs[i], 
                config.dis_j_fcs[i + 1],
                dropout_rate=self.do_rate, 
                activation_type='relu') 
                for i in range(len(config.dis_j_fcs) - 2)]
        )
        self.joint_part.append(FCBlock(
            config.dis_j_fcs[len(config.dis_j_fcs) - 2], 
            config.dis_j_fcs[len(config.dis_j_fcs) - 1],
            dropout_rate=self.do_rate, 
            activation_type='sigmoid'
        ))

    def forward(self, x, y):
        for img_emb_layer in self.img_part:
            x = img_emb_layer(x)

        for y_emb_layer in self.y_part:
            y = y_emb_layer(y)

        j_emb = torch.cat((x, y), dim=1)
        
        for dis_layer in self.joint_part:
            j_emb = dis_layer(j_emb)

        return j_emb