import torch.nn as nn


class Thinking(nn.Module):
    def __init__(self, config_dict={}):
        super(Thinking, self).__init__()

        self.config = config_dict

        self.output_dim = self.config.get("output_dim", 2)
        self.channel_dim = self.config.get("channel_dim", 4)

        output_dim = self.output_dim

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.channel_dim, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True)
        )

        self.fc1 = nn.Sequential(nn.Linear(7 * 7 * 64, 512), nn.ReLU(inplace=True))
        self.fc2 = nn.Linear(512, output_dim)
        # self._create_weights()

        self.feal_criterion = nn.CosineSimilarity(dim=1)

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.01, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):  # (1, 4, 84, 84)
        output = self.conv1(input)  # (1, 32, 20, 20)
        output = self.conv2(output)  # (1, 64, 9, 9)
        output = self.conv3(output)  # (1, 64, 7, 7)
        output = output.view(output.size(0), -1)  # (B, 3136)
        output = self.fc1(output)  # (1, 512)
        output = self.fc2(output)  # (1, 2)

        return output
