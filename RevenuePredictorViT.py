import torch
import torch.nn as nn
from dino import vision_transformer as vits

class RevenuePredictorViT(nn.Module):
    def __init__(self, args, args_vit, args_dnn):
        super().__init__()

        student, teacher = self.load_pretrained_ViT(args_vit)

        self.vit_student = student
        self.vit_teacher = teacher

        self.vit_output_size = \
            args['last_n_blocks'] * self.vit_teacher.embed_dim

        self.vit2dnn = nn.Linear(self.vit_output_size, args['n_dnn_img_features'])
        self.dnn = ...


        self.args = args

    def forward(self, img, features):
        # vit
        with torch.no_grad(): # TODO finetune ViT, is a bit complex
            model = self.vit_teacher
            intermediate_output = model.get_intermediate_layers(img, self.args['last_n_blocks'])
            vit_output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
            if self.args['avgpool']:
                vit_output = torch.cat((vit_output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                vit_output = vit_output.reshape(vit_output.shape[0], -1)

        # vit2dnn
        fc_output = self.vit2dnn(vit_output)

        # dnn
        dnn_input = torch.cat(fc_output, features) # concatenate img features with other movie details
        output = self.dnn(dnn_input)
        return output

    # create function to load pretrained model
    def load_pretrained_ViT(self, args):

        # initialize models
        student = vits.__dict__[args['arch']](patch_size=args['patch_size'], num_classes=0)
        teacher = vits.__dict__[args['arch']](patch_size=args['patch_size'], num_classes=0)

        # fetch pretrained models
        url = None
        if args['arch'] == "vit_small" and args['patch_size'] == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_full_checkpoint.pth"
        elif args['arch'] == "vit_small" and args['patch_size'] == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain_full_checkpoint.pth"  # model used for visualizations in our paper
        elif args['arch'] == "vit_base" and args['patch_size'] == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain_full_checkpoint.pth"
        elif args['arch'] == "vit_base" and args['patch_size'] == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain_full_checkpoint.pth"

        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)

        state_dict_teacher = state_dict['teacher']
        # remove `module.` prefix
        state_dict_teacher = {k.replace("module.", ""): v for k, v in state_dict_teacher.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict_teacher = {k.replace("backbone.", ""): v for k, v in state_dict_teacher.items()}

        msg = teacher.load_state_dict(state_dict_teacher, strict=False)
        print('Pretrained weights found and loaded with msg: {}'.format(msg))

        state_dict_student = state_dict['student']
        # remove `module.` prefix
        state_dict_student = {k.replace("module.", ""): v for k, v in state_dict_student.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict_student = {k.replace("backbone.", ""): v for k, v in state_dict_student.items()}

        msg = student.load_state_dict(state_dict_student, strict=False)
        print('Pretrained weights found and loaded with msg: {}'.format(msg))

        # test this by running the eval thing
        return student, teacher