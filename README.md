#Installation:
必須模組
1. Pytorch >=1.8.0 

可至下方Potorch網站複製命令至CMD下載，版本務必1.8.0以上
https://pytorch.org/get-started/previous-versions/

2. detectron2 

於CMD中執行下列命令
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
若失敗多次可參考下方網址
https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md

3. opencv

pip install opencv-python

4. fvcore

pip install fvcore


#Train:
程式碼我們參考了CenterNetv2的官方程式
https://github.com/xingyizhou/CenterNet2/blob/master/README.md
執行 CenterNet2/train_net.py 即可訓練
train_net.py提供四種參數修改


NUM_CLASSES (類別數量)
args.config_file (參數檔yaml的路徑)
register_coco_instances (訓練與測試集)

#Configs:
格式為yaml，至configs資料夾中選取
CenterNet2_R2-101-DCN_896_4x.yaml與CenterNet2_R2-101-DCN-BiFPN_1280_4x.yaml
皆是基於Base-CenterNet2.yaml

#Weight:
我們使用的骨幹網路為r2_101.pkl (res2net101)
訓練時，yaml檔中的WEIGHT需使用r2_101.pkl的路徑
另一份權重best_model_res2net101_FPN_896為我們的最佳方法

#Demo:
執行 CenterNet2/demo.py 即可測試
提供六種參數修改

1. config-file (使用的參數檔路徑)
2. input (輸入的圖像資料夾路徑)
3. output (輸出結果的資料夾路徑，若無該資料夾，將逐一顯示視窗)
4. confidence-threshold (信心分數)
5. opts (權重檔的路徑)
6. txtpath (輸出官方指定json檔的路徑與名稱)

#Data:
我們提供了兩種數據

2label_stas.mat (兩種樣本的matlab標註數據)
lung_obj (我們使用的coco格式數據)

