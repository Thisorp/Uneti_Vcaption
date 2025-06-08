# Thực hiện sinh chú giải ảnh bằng mô hình EN-DE, C[aA]RNet!   
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)   ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)  [![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)  

**Convolutional(and|Attention)RecurrentNet!**
Mục tiêu của dự án là xây dựng một mô hình Neural để sinh chú giải cho ảnh.

Với một bộ dữ liệu, một mạng thần kinh gồm:
- Bộ mã hóa (Encoder - Mạng Neural Residual đã được huấn luyện trước)
- Bộ giải mã (Decoder - Mô hình LSTM)
Sẽ biểu diễn ảnh trong một không gian do bộ mã hóa xác định, biểu diễn này được đưa vào bộ giải mã (theo nhiều cách khác nhau) để học sinh ra chú thích,
liên kết với ảnh và các từ đã sinh ra ở mỗi bước thời gian bởi LSTM.

Bạn có thể xem bảng mục lục dưới đây:


Trong thư mục ẩn `.saved` có thể tìm thấy tất cả các phiên bản đã huấn luyện của C[aA]RNet.

# Mục lục
- [Một hiện thực Pythonic cho Bài toán Chú thích Ảnh, C[aA]RNet!](#thuc-hien-sinh-chu-giai-anh-bang-mo-hinh-en-de-caarnet)
- [Mục lục](#muc-luc)
- [Kiến thức tiền đề](#kien-thuc-tien-de)
- [Cách chạy mã nguồn](#cach-chay-ma-nguon)
  * [Các phiên bản Python hỗ trợ](#cac-phien-ban-python-ho-tro)
  * [Thư viện phụ thuộc](#thu-vien-phu-thuoc)
  * [Biến môi trường](#bien-moi-truong)
  * [Giải thích CLI](#giai-thich-cli)
    * [Ví dụ](#vi-du)
  * [Tích hợp GPU](#tich-hop-gpu)
- [Pipeline Dữ liệu](#pipeline-du-lieu)
  * [Định dạng Dataset](#dinh-dang-dataset)
    + [Ảnh](#anh)
    + [Kết quả](#ket-qua)
- [Script sinh ra những gì](#script-sinh-ra-nhung-gi)
  * [Trong quá trình huấn luyện](#trong-qua-trinh-huan-luyen)
  * [Trong quá trình đánh giá](#trong-qua-trinh-danh-gia)
- [Cấu trúc dự án](#cau-truc-du-an)
  * [Filesystem](#filesystem)
  * [Interfaces](#interfaces)
  * [Encoder](#encoder)
    + [CResNet50](#cresnet50)
    + [CResNet50Attention](#cresnet50attention)
  * [Decoder](#decoder)
    + [RNetvI](#rnetvi)
    + [RNetvH](#rnetvh)
    + [RNetvHC](#rnetvhc)
    + [RNetvHCAttention](#rnetvhcattention)
- [Quy trình huấn luyện](#quy-trinh-huan-luyen)
  * [Loại hàm mất mát](#loai-ham-mat-mat)
    + [Lưu ý: Mất mát trong phiên bản attention](#luu-y-mat-mat-trong-phien-ban-attention)
- [Thí nghiệm cá nhân](#thi-nghiem-ca-nhan)
- [Tài liệu tham khảo](#tai-lieu-tham-khao)
  * [Tác giả](#tac-gia)

# Kiến thức tiền đề
Để hiểu rõ hơn về mã nguồn và thông tin bên trong, vì repo này hướng tới việc dễ hiểu cho cả người mới tò mò lẫn người đã có chuyên môn, bạn nên tham khảo các tài liệu sau:

-[Tài liệu Pytorch](https://pytorch.org/docs/stable/index.html)
-[Mạng Neural Tích chập (Stanford Edu)](https://cs231n.github.io/convolutional-networks/)
-[Mạng Neural Hồi tiếp (Stanford Edu)](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks#architecture)
-[Mạng Neural Residual (D2L AI)](https://d2l.ai/chapter_convolutional-modern/resnet.html)


# Cách chạy mã nguồn
[![Linux](https://svgshare.com/i/Zhy.svg)](https://svgshare.com/i/Zhy.svg) [![macOS](https://svgshare.com/i/ZjP.svg)](https://svgshare.com/i/ZjP.svg) [![Windows](https://svgshare.com/i/ZhY.svg)](https://svgshare.com/i/ZhY.svg)

Mã nguồn có thể chạy trên mọi hệ điều hành, bạn có thể dùng bất kỳ hệ điều hành nào. Tuy nhiên, một máy cấu hình cao là bắt buộc, vì lượng dữ liệu lớn có thể gây lỗi hết bộ nhớ trên máy yếu.
**Lưu ý rằng bạn cần tải bộ dữ liệu trước khi chạy và nó phải đúng định dạng yêu cầu**
Cách chuẩn bị dataset cho huấn luyện C[aA]RNet?

 1. [Tải về](https://drive.google.com/file/d/1Y9nuHtmO0p0Jd2lvzIF5euiemkJHJ2IU/view?usp=sharing) bộ dữ liệu đã được vietsub từ Kaggle Flickr30k.
 2. Giải nén vào thư mục gốc của repo.
 3. Đổi tên thư mục thành *dataset*
 4. Đổi tên thư mục ảnh thành *images*

Nếu bạn có trường hợp đặc biệt, có thể chỉnh file [VARIABLE.py](#bien-moi-truong) và/hoặc một số tham số tùy chọn trước khi chạy script ([Giải thích CLI](#giai-thich-cli)
).

## Các phiên bản Python hỗ trợ
Mã nguồn sẵn sàng chạy với mọi phiên bản python lớn hơn 3.6.
Như bạn sẽ thấy trong mã, một số tiện ích không có ở python <3.9. Các trường hợp này đều có chú thích trong mã, bạn có thể chọn bật/tắt theo ý muốn.

## Thư viện phụ thuộc
| Thư viện | Phiên bản  |
| ------------ | ------------ |
|  Torch | 1.3.0+cu100  |
|  Torchvision | 0.4.1+cu100  |
|  Pillow | 8.4.0  |
|  Numpy | 1.19.5  |
|  Pandas | 1.1.5  |
|  Matplotlib | 3.3.4  |

Trong thư mục gốc có file requirements.txt, bạn có thể cài tất cả các package cần thiết vào môi trường (hoặc v.env.) bằng lệnh sau, chạy trong shell với môi trường đã kích hoạt:
```bash
pip install -r requirements.txt
```

Nếu pip không tìm thấy đúng phiên bản torch, bạn có thể chạy lệnh sau với venv đã kích hoạt:
```bash
pip install torch==1.3.0+cu100 torchvision==0.4.1+cu100 -f https://download.pytorch.org/whl/torch_stable.html
```

## Biến môi trường
Vì một số thuộc tính của repo được dùng ở nhiều file, tạo một container môi trường là cách hợp lý.
Dùng file `.env` là cách đơn giản nhất, nhưng để đảm bảo tương thích đa nền tảng, file `VARIABLE.py` là giải pháp tốt.
Các HẰNG SỐ được định nghĩa như sau:
|HẰNG SỐ| Ý NGHĨA |
|--|--|
| MAX_CAPTION_LENGTH | Chỉ lấy các mẫu có chú thích ngắn hơn hoặc bằng giá trị này |
|IMAGES_SUBDIRECTORY_NAME| Tên thư mục chứa ảnh (phải nằm trong thư mục dataset) | 
| CAPTION_FILE_NAME | Tên file chứa tất cả caption, nằm ở thư mục dataset.|
| EMBEDDINGS_REPRESENTATION | Cách tạo word embedding. CHƯA DÙNG |

## Giải thích CLI
Mã nguồn chạy qua shell, dưới đây là hướng dẫn chạy đúng script, các tham số tùy chỉnh và ý nghĩa của chúng.
Phần **luôn luôn có** là gọi trình thông dịch và file chính:
```bash
python main.py
```
Sau đó, helper sẽ hiện ra như sau:
```bash
usage: main.py [-h] [--attention ATTENTION]
               [--attention_dim ATTENTION_DIM]
               [--dataset_folder DATASET_FOLDER]
               [--image_path IMAGE_PATH]
               [--splits SPLITS [SPLITS ...]]
               [--batch_size BATCH_SIZE]
               [--epochs EPOCHS] [--lr LR]
               [--workers WORKERS]
               [--device DEVICE]
               {RNetvI,RNetvH,RNetvHC,RNetvHCAttention}
               {train,eval} encoder_dim hidden_dim
```

Phần bắt buộc gồm các tham số:
| Tham số  | Ý nghĩa  | Hành vi đặc biệt  |
| ------------ | ------------ | ------------ |
| decoder| Bộ giải mã bạn muốn dùng, các lựa chọn: {RNetvI,RNetvH,RNetvHC,RNetvHCAttention} | Mô tả từng loại decoder ở các phần sau  |
| mode  | Chế độ hoạt động, train để huấn luyện, eval để đánh giá  | lựa chọn: {train,eval}  |
| encoder_dim  | Kích thước chiếu ảnh của encoder  |  Decoder=RNetvI => Không quan tâm / Decoder=RNetvHCAttention => 2048  |
| hidden_dim  | Sức chứa của LSTM  |   |

Tham số tùy chọn:
| Tham số | Ý nghĩa | Mặc định |
|--|--|--|
| --attention | Dùng attention. Mặc định False | Nếu bật, decoder và encoder sẽ là CResNet50Attention và RNetvHCAttention |
| --attention_dim | Sức chứa của attention unit. (Mặc định 1024)||
| --dataset_folder | Thư mục chứa dữ liệu. (Mặc định "./dataset")| Chỉ dùng khi train|
| --image_path | Đường dẫn ảnh cần sinh caption. (Mặc định '') | Chỉ dùng khi eval |
| --splits | Tỷ lệ train, val, test (Mặc định: 60 30 10) | Chỉ dùng khi train|
| --batch_size | Kích thước mini batch (Mặc định: 32) | Chỉ dùng khi train |
| --epochs | Số epoch huấn luyện (Mặc định: 500) | Chỉ dùng khi train |
| --lr | Learning rate (Adam) (mặc định: 1e-3) | Chỉ dùng khi train |
| --workers | Số worker load dữ liệu (Mặc định: 4) | Chỉ dùng khi train |
| --device| Thiết bị tính toán \in {cpu, cuda:0, cuda:1, ...} (Mặc định: cpu) | Chỉ dùng khi train |

### Ví dụ
Các ví dụ sau là lệnh tôi dùng cho các thí nghiệm cá nhân.

**Huấn luyện**

`CaRNetvI`
```bash
python main.py RNetvI train 0 1024 --dataset_folder ./dataset --device cuda:0 --epochs 150
```
`CaRNetvH`
```bash
python main.py RNetvH train 1024 1024 --dataset_folder ./dataset --device cuda:0 --epochs 150
```
`CaRNetvHC`
```bash
python main.py RNetvHC train 1024 1024 --dataset_folder ./dataset --device cuda:0 --epochs 150
```
`CaRNetvHCAttention`
```bash
python main.py RNetvHCAttention train 1024 1024 --dataset_folder ./dataset --device cuda:0 --epochs 150 --attention t --attention_dim 1024
```

**Đánh giá**

`CaRNetvI`
```bash
python main RNetvI eval 5078 1024 --image_path ./33465647.jpg
```
`CaRNetvH`
```bash
python main.py RNetvH eval 1024 1024 --image_path ./33465647.jpg
```
`CaRNetvHC`
```bash
python main.py RNetvHC eval 1024 1024 --image_path ./33465647.jpg
```
`CaRNetvHCAttention`
```bash
python main.py RNetvHCAttention  eval 1024 1024 --attention t --attention_dim 1024 --image_path ./33465647.jpg
```
**Đánh giá toàn bộ thư mục ảnh (Folder Evaluation)**

```bash
python eval.py RNetvHCAttention eval 1024 1024 --attention t --attention_dim 1024 --dataset_folder ./testset --output_csv ./testset/caption_test4.csv
```

> 📄 File `results.csv` sẽ được tạo trong đường dẫn `--output_csv`, theo định dạng: `image_name| comment_number| comment`.

**📊 Đánh giá độ chính xác BLEU**

```bash
python bleu_newest.py ./caption_test.csv ./results.csv
```

> So sánh giữa nhãn thực tế (`caption_test.csv`) và dự đoán từ mô hình (`results.csv`).

---

📁 **Chú ý cấu trúc thư mục:**
- `./dataset/` hoặc `./testset/`
  - `images/` (thư mục chứa ảnh)
  - `results.csv` (file chú thích, phân tách bằng dấu `|`)

---

## Tích hợp GPU

Như bạn đã thấy ở phần CLI, mã nguồn hỗ trợ GPU (chỉ NVIDIA hiện tại).
Bạn cần cài driver CUDA, để đồng bộ với torch trong requirements.txt và driver, bạn nên cài NVIDIA driver v440 + Cuda 10.2.

# Pipeline Dữ liệu

Để hiểu rõ hơn về những gì xảy ra trong script, hãy hình dung pipeline dữ liệu như sau:

- Dataset: chứa tất cả các ví dụ (chưa tách train/test/val)
- Vocabulary: mỗi ví dụ có caption, nên cần một từ điển chứa tất cả các từ.
- C[aA]RNet: mạng neural, chưa phân biệt có/không Attention. 

Giải thích chi tiết về từng thành phần ở các phần sau.
Hiện tại chỉ cần biết script cần 3 thành phần này để làm việc với dữ liệu.
![Data Pipeline](https://i.imgur.com/d8OtmUu.png)
Hãy tưởng tượng mỗi thao tác là một bước thời gian.
- T_0: Load dataset
- T_1:
  - a) Dataset được chuyển thành Dataloader (lớp của pytorch).
  - b) Tạo từ điển từ dataset.
 - T_2: Tạo dataloader
 - T_3: C[aA]RNet dùng cả dataloader và vocabulary để huấn luyện, khi đánh giá chỉ dùng vocabulary vì dataloader size 1.
 
## Định dạng Dataset

Cách định nghĩa Dataset theo cấu trúc của bộ Flickr30k Image Dataset: https://www.kaggle.com/hsankesara/flickr-image-dataset

Cấu trúc filesystem như sau:

dataset/
├─ images/
│  ├─ pippo_pluto_paperino.jpg
├─ results.csv

 ### Ảnh
Thư mục images chứa các ảnh jpeg, tên không có dấu cách.
`pippo_pluto_paperino.jpg`

### Kết quả
File chứa các caption.
**Vì** caption có thể chứa dấu phẩy *(,)* , nên ký tự phân tách là dấu gạch đứng *(|)*.
Dòng đầu là header, các cột như sau:
| Tham số | Kiểu     | Mô tả                       |
| :-------- | :------- | :-------------------------------- |
| `image_name`      | `string` | Tên file ảnh tương ứng |
| `comment_number`      | `int` | Chỉ số của caption |
| `comment`*      | `string` | Caption |

*Caption nên tách từ bằng dấu cách.
Dấu chấm (".") đánh dấu kết thúc caption.

# Script sinh ra những gì
Vì dự án còn phát triển tiếp, nên cần mô tả các output chính:

- Output trong quá trình huấn luyện.
- Output khi đánh giá.

## Trong quá trình huấn luyện
Sinh ra các output sau:

 1. Mỗi mini-batch của mỗi epoch: lưu loss và accuracy vào Dataframe.
 2. Mỗi epoch, lưu accuracy trên tập validation vào Dataframe.
 3. Mỗi epoch, lưu một caption sinh ra từ ảnh cuối cùng của batch cuối cùng trong validation.
 4. Mỗi khi đạt accuracy tốt nhất trên validation, lưu model vào bộ nhớ.

### 1
Dataframe lưu thành file *train_results.csv* cuối mỗi epoch, cấu trúc:
| Tham số | Kiểu     | Mô tả                       |
| :-------- | :------- | :-------------------------------- |
|   Epoch   | `int` | ID của epoch |
|   Batch   | `int` | ID của batch |
|   Loss    | `float` | Loss của batch|
|   Accuracy    | `float` | Accuracy của batch |

### 2
Dataframe lưu thành file *validation_results.csv* cuối mỗi epoch, cấu trúc:
| Tham số | Kiểu     | Mô tả                       |
| :-------- | :------- | :-------------------------------- |
|   Epoch   | `int` | ID của epoch |
|   Accuracy    | `float` | Accuracy trên validation|

### 3
Trích đặc trưng từ ảnh cuối của batch cuối validation, đưa vào net ở chế độ eval.
Sinh file caption.png gồm caption sinh ra và ảnh gốc.
Nếu có attention, sinh thêm attention.png thể hiện attention cho từng từ.

### 4
Mỗi khi đạt accuracy tốt nhất trên validation, lưu model vào bộ nhớ.
Thư mục lưu là `.saved` ở gốc repo.
Pattern file:

 - Encoder: NetName_encoderdim_hiddendim_attentiondim_C,pth
 - Decoder: NetName_encoderdim_hiddendim_attentiondim_R,pth
 
Các tham số này phụ thuộc vào cấu hình khi huấn luyện.

## Trong quá trình đánh giá
Ảnh được load, tiền xử lý, đưa vào C[aA]RNet.
Sinh file caption.png gồm caption sinh ra và ảnh gốc.
Nếu có attention, sinh thêm attention.png thể hiện attention cho từng từ.

# Cấu trúc dự án
Cấu trúc dự án tính đến khả năng mở rộng từ cộng đồng hoặc cá nhân.
Sơ đồ dưới đây chỉ mang tính tổng quát, thể hiện các thực thể và quan hệ phụ thuộc.
Mỗi phương thức đều có docstring, hãy dùng làm tài liệu tham khảo.
![UML](https://i.imgur.com/xmGekz5.jpg)

## Filesystem
Cấu trúc filesystem như sau:

    C[aA]RNet/
    ├─ .saved/
    ├─ dataset/
    │  ├─ images/
    │  ├─ results.csv
    ├─ NeuralModels/
    │  ├─ Attention/
    │  │  ├─ IAttention.py
    │  │  ├─ SoftAttention.py
    │  ├─ Decoder/
    │  │  ├─ IDecoder.py
    │  │  ├─ RNetvH.py
    │  │  ├─ RNetvHC.py
    │  │  ├─ RNetvHCAttention.py
    │  │  ├─ RNetvI.py
    │  ├─ Encoder/
    │  │  ├─ IEncoder.py
    │  │  ├─ CResNet50.py
    │  │  ├─ CResNet50Attention.py
    │  ├─ CaARNet.py
    │  ├─ Dataset.py
    │  ├─ FactoryModels.py
    │  ├─ Metrics.py
    │  ├─ Vocabulary.py
    ├─ VARIABLE.py
    ├─ main.py
 
| File | Mô tả |
|--|--|
| `VARIABLE.py` | Giá trị hằng dùng trong dự án|
| `main.py` | Điểm vào để chạy net|
| `IAttention.py` | Interface cho attention mới |
| `SoftAttention.py` | Hiện thực Soft Attention |
| `IDecoder.py` | Interface cho decoder mới |
| `RNetvH.py` | Hiện thực decoder LSTM H-version |
| `RNetvHC.py` | Hiện thực decoder LSTM HC-version |
| `RNetvHCAttention.py` | Hiện thực decoder LSTM HC-version với Attention|
| `IEncoder.py` | Interface cho encoder mới |
| `CResNet50.py` | ResNet50 làm encoder |
| `CResNet50Attention.py` | ResNet50 cho attention |
| `CaRNet.py` | Hiện thực C[aA]RNet |
| `Dataset.py` |  Quản lý dataset |
| `FactoryModels.py` | Factory Pattern cho các mô hình |
| `Metrics.py` | Sinh file báo cáo |
| `Vocabulary.py` | Quản lý từ điển |


## Interfaces
Interface dùng để định nghĩa hợp đồng cho ai muốn hiện thực Encoder, Decoder hoặc Attention mới.
Tuân thủ interface là bắt buộc, docstring có gợi ý tham số cho từng phương thức.

## Encoder
Hai encoder dựa trên ResNet50 *(He et al. 2015, Deep Residual Learning for Image Recognition)*.
Tùy có dùng attention hay không, sẽ bỏ một hoặc nhiều lớp cuối của net gốc.

![ResNet50](https://www.researchgate.net/publication/336805103/figure/fig4/AS:817882309079050@1572009746601/ResNet-50-neural-network-architecture-56.ppm)

(Kiến trúc mạng ResNet-50 [56].) [Privacy-Constrained Biometric System for Non-Cooperative Users](https://www.researchgate.net/publication/336805103_Privacy-Constrained_Biometric_System_for_Non-Cooperative_Users)

### CResNet50
Bản 1 bỏ lớp cuối của ResNet50, để lộ GlobalAveragePooling. Sau pooling là một lớp tuyến tính kích thước *encoder_dim*, nhận đầu vào là output của AveragePooling (ResNet50 là 2048).
 
### CResNet50Attention
Bản 2 bỏ 2 lớp cuối của ResNet50 (AveragePooling + FC), để lộ lớp tích chập cuối cho tensor dạng: (Heigth/32, Width/32, 2048). 
Mỗi vùng là một vector 2048 chiều. 
Với ảnh RGB vuông (3,224,224) thì tổng số vùng là 49.

## Decoder
Decoder dựa trên RNN, cụ thể là LSTM (Long-Short Term Memory), một loại RNN cập nhật trạng thái ẩn đặc biệt.
![LSTM](https://www.researchgate.net/profile/Xuan_Hien_Le2/publication/334268507/figure/fig8/AS:788364231987201@1564972088814/The-structure-of-the-Long-Short-Term-Memory-LSTM-neural-network-Reproduced-from-Yan.png)
*(Cấu trúc LSTM. Tái bản từ Yan [38].)* [Ứng dụng LSTM cho dự báo lũ](https://www.researchgate.net/publication/334268507_Application_of_Long_Short-Term_Memory_LSTM_Neural_Network_for_Flood_Forecasting)

Mỗi mô hình xuất phát từ ý tưởng này và thử các cách khác nhau để đưa context ảnh từ encoder vào:

 1. RNetvI: Context ảnh là input đầu tiên của LSTM tại t_0.
 2. RNetvH: Context ảnh được đưa vào hidden state tại t_0.
 3. RNetvHC: Context ảnh đưa vào cả hidden và cell state tại t_0.
 4. RNetvHCAttention: Context ảnh đưa vào hidden và cell state, mỗi bước t nối thêm vector attention vào input LSTM. 
 
### RNetvI
Bản 1 dùng context ảnh làm input đầu tiên của lstm.

![RNetvI](https://i.imgur.com/PAxWnQy.png)
(Vinyals et al. 2014) [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555) 

Ràng buộc duy nhất là context ảnh phải chiếu vào không gian embedding từ.

### RNetvH
RNetvH khởi tạo hidden state tại t_0 bằng context ảnh từ ResNet.

![RNetvH](https://i.imgur.com/9b2vVt3.jpg)
(Vinyals et al. 2014) [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555) (Bản chỉnh sửa bởi christiandimaio)
### RNetvHC
RNetvHC khởi tạo cả hidden và cell state tại t_0 bằng context ảnh từ ResNet
![RNetvHC](https://i.imgur.com/pCrj3TS.jpg)
(Vinyals et al. 2014) [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555) (Bản chỉnh sửa bởi christiandimaio)

### RNetvHCAttention
Bản này kết hợp RNetvHC với Attention.
![RNetvHCAttention](https://i.imgur.com/64rTN7q.png)
Credit to christiandimaio et al. 2022

# Quy trình huấn luyện
Quy trình huấn luyện gồm tập train và tập validation.

 - Tập train được chia thành các mini-batch (tham số) và xáo trộn.
	 - Với mỗi mini-batch: 
		 - Đưa batch vào encoder để sinh context vector cho từng phần tử.
		 - Giả sử tensor caption (đã chuyển thành vector id từ vocabulary) của batch ảnh được padding bằng 0 và sắp xếp giảm dần theo độ dài.
		 - Context vector và caption được đưa vào Decoder.
		 - Output decoder là input cho pack_padded_sequence, loại bỏ vùng pad của mỗi caption.
		 - Tính loss, backpropagation và cập nhật trọng số.
 - Đánh giá accuracy trên tập validation.
	 - Nếu có model tốt nhất mới, lưu lại model.

## Loại hàm mất mát
Hàm mất mát dùng là CrossEntropyLoss, vì pytorch nội bộ dùng soft-max trên mỗi output t (output lstm có kích thước bằng vocab, ta muốn chọn từ xác suất cao nhất) và NegativeLogLikelihood.
<p align="center">
  <img src="https://i.imgur.com/PBZbhjR.png" />
</p>
Với p_t:
<p align="center">
  <img src="https://i.imgur.com/iz2a86l.png" />
</p>

Hàm mất mát theo paper (Vinyals et al. 2014) [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/search/cs?searchtype=author&query=Vinyals%2C+O)
### Lưu ý: Mất mát trong phiên bản attention
Với attention, thêm một thành phần vào loss: double stochastic regularization.
<p align="center">
  <img src="https://i.imgur.com/mNbrTo5.png" />
</p>
Điều này khuyến khích model chú ý đều đến mọi phần của ảnh trong quá trình sinh caption.

## Thí nghiệm cá nhân
Dưới đây là các lần huấn luyện tôi đã thực hiện, các model pretrained nằm trong thư mục `.saved`
![Training Table](https://i.imgur.com/sqgEPzM.png)

# Tài liệu tham khảo

 -  (Vinyals et al. 2014) [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555) 
 - (Xu et al. 2015) [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/pdf/1502.03044.pdf)
 - Tài liệu, mã nguồn và bài giảng của Giáo sư [Stefano Melacci](https://www3.diism.unisi.it/~melacci/)

## 📑 Tóm tắt file eval_app.py

`eval_app.py` là ứng dụng web sử dụng Streamlit để đánh giá mô hình sinh chú thích ảnh tiếng Việt với các mô hình CaRNet. Ứng dụng hỗ trợ hai chế độ:
- **Ảnh đơn**: Cho phép người dùng tải lên một ảnh, sinh caption và hiển thị kết quả trực tiếp.
- **Toàn bộ thư mục**: Đánh giá tự động tất cả ảnh trong một thư mục, xuất kết quả ra file CSV.

Các tính năng chính:
- Chọn mô hình CaRNet (vI, vH, vHC, vHCAttention) và giao diện sáng/tối.
- Tùy chọn chế độ đánh giá (ảnh đơn hoặc thư mục).
- Hiển thị logo, tiêu đề, mô tả ứng dụng.
- Tự động tải và cấu hình mô hình phù hợp.
- Tính điểm BLEU cho caption.
- Lưu kết quả đánh giá hàng loạt vào file `eval_results.csv`.

Ứng dụng giúp kiểm thử, trình diễn và đánh giá nhanh hiệu quả các mô hình sinh chú thích ảnh trong dự án.

## Tác giả

- [@Thisorp](https://www.github.com/Thisorp)