import os
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch.utils.data import Dataset
import torch
from timm.utils import accuracy, AverageMeter
from torch import nn
from torchvision import transforms
from collections import defaultdict
from torch.utils.data import Dataset
from model import mscet
from PIL import Image

valdir = r'F:\岩矿数据\大类\data10-24\val'
batch_size = 32
num_cls = 5
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
val_data_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize])
model = mscet()
state_dict = torch.load('checkpoint_best.pth')
model.head = nn.Linear(in_features=1280, out_features=num_cls, bias=True)
model.load_state_dict(state_dict['model'])
model = model.cuda()


class GroupedImagesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.class_to_idx = self._get_class_to_idx()
        self.groups, self.labels = self._group_images_by_prefix()

    def _get_class_to_idx(self):
        class_to_idx = {}
        for idx, class_name in enumerate(sorted(os.listdir(self.root_dir))):
            class_to_idx[class_name] = idx
        return class_to_idx

    def _group_images_by_prefix(self):
        grouped_images = defaultdict(list)
        labels = {}
        for class_name in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_path):
                label = self.class_to_idx[class_name]
                for filename in os.listdir(class_path):
                    if '&' in filename:
                        prefix = filename.split('&')[0] + '&' + filename.split('&')[1].split('_')[0]
                    else:
                        prefix = filename.split('+')[0] + '+' + filename.split('+')[1]
                    group_key = f"{class_name}_{prefix}"
                    file_path = os.path.join(class_path, filename)
                    grouped_images[group_key].append(file_path)
                    labels[group_key] = label
        return grouped_images, labels

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):

        group_key = list(self.groups.keys())[idx]
        classname = group_key.split('_')[0]
        label_list = {'Mica': 0, 'Detritus': 1, 'Flint': 2, 'Quartz': 3, 'Feldspar': 4}
        label = label_list[classname]
        image_paths = self.groups[group_key]

        images = [self.transform(Image.open(img_path).convert('RGB')) for img_path in image_paths]
        return torch.stack(images), torch.tensor(label, dtype=torch.long), image_paths


# 示例：创建数据集和数据加载器
dataset = GroupedImagesDataset(root_dir=valdir, transform=val_data_transforms)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

criterion = torch.nn.CrossEntropyLoss()
model.eval()
losses = AverageMeter()
top1 = AverageMeter()
class_acc = [AverageMeter() for _ in range(5)]


all_final_predictions = []
all_true_labels = []
model.eval()
correct_predictions = 0
total_predictions = 0

with torch.no_grad():
    for batch_idx, (image_group, targets, image_path) in enumerate(data_loader):
        targets = targets.cuda()
        for group_idx, group in enumerate(image_group):
            acc = 0
            total = 0
            final_acc = 0
            final_total = 0
            final=0
            ratio_flag = 0
            group_target = targets[group_idx].item()
            target = targets[group_idx]
            softmax_scores = []
            original_predictions = []

            for i, image in enumerate(group):
                image = image.unsqueeze(0).cuda()
                pred = model(image)
                softmax = torch.nn.functional.softmax(pred, dim=1)
                softmax_scores.append(softmax)
                original_predictions.append(pred.argmax(dim=1))
            for original in original_predictions:
                if original == group_target:
                    acc += 1
                total += 1
            orginal_acc = acc / total if acc > 0 else 0
            max_score = 0
            max_score_class = None
            for softmax in softmax_scores:
                top_score = softmax.max()
                if top_score > max_score:
                    max_score = top_score
                    max_score_class = softmax.argmax().item()
            final_predictions = []
            for i, softmax in enumerate(softmax_scores):
                top_scores, _ = torch.topk(softmax, 2)
                ratio = top_scores[0][0] / (top_scores[0][0] + top_scores[0][1])
                print(image_path[i], ratio, original_predictions[i].item())
                if ratio < 0.6:
                    final_predictions.append(max_score_class)
                else:
                    final_predictions.append(original_predictions[i].item())
            for final_pred in final_predictions:
                if final_pred == group_target:
                    final_acc +=1
                final_total+=1
            final = final_acc/final_total if final_acc>0 else 0
            if ratio_flag and orginal_acc != final :
                print(orginal_acc,final,image_path)
            all_final_predictions.extend(final_predictions)
            all_true_labels.extend([group_target] * len(final_predictions))
            for pred_label in final_predictions:
                if pred_label == group_target:
                    correct_predictions += 1
                total_predictions += 1

accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
print(f'Accuracy: {accuracy:.4f}')
precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, all_final_predictions, average=None,
                                                           labels=range(5))
overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(all_true_labels,
                                                                                   all_final_predictions,
                                                                                   average='macro')
overall_accuracy = accuracy_score(all_true_labels, all_final_predictions)