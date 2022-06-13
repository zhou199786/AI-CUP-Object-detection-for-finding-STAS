addpath('C:\Program Files\MATLAB\R2021b\toolbox\jsonlab-master');    
jsondata1= loadjson("D:\Competition\lung\coco_instances_results_2.json");
jsondata = loadjson("D:\Competition\lung\coco_instances_results_1.json");

imgdir = dir("D:\Competition\lung\OBJ\Train_Images\*.jpg");
imgdir = {imgdir.name};
img_name = string(imgdir);
dataSource = groundTruthDataSource(img_name);

ldc = labelDefinitionCreator();
addLabel(ldc,'stas',labelType.Rectangle);
addLabel(ldc,'nostas',labelType.Rectangle);
labelDefs = create(ldc);

numlabel = 1
imgid = 0;
STAS = [];
box_matirx = [];
for i =1:length(jsondata)
    if jsondata(i).image_id == imgid
        box_matirx = [box_matirx;jsondata(i).bbox];
        
        STAS{numlabel,1} = box_matirx;
        imgid = jsondata(i).image_id;
    else
        numlabel = numlabel+1;
        box_matirx = [];
        box_matirx = [box_matirx;jsondata(i).bbox];
        STAS{numlabel,1} = box_matirx;
        
        imgid = jsondata(i).image_id;
    end
      
end

imgid1 = 527;

box_matirx = [];
for i =1:length(jsondata1)
    if jsondata1(i).image_id == imgid1

        box_matirx = [box_matirx;jsondata1(i).bbox];
        STAS{numlabel,1} = box_matirx;
        imgid1 = jsondata1(i).image_id;
    else
        
        if imgid1~=(numlabel-1)
            STAS{numlabel,1} = [];
            numlabel = numlabel+1;
        else
            numlabel = numlabel+1;
            box_matirx = [];
            box_matirx = [box_matirx;jsondata1(i).bbox];
            STAS{numlabel,1} = box_matirx;
            
            imgid1 = jsondata1(i).image_id;
        end
    end
end


nostas = [];
for k =1:length(STAS)
    iou =[];
    box = gTruth.LabelData.stas{k,1};
    for bbox = 1:size(box,1)
        nobox = [];
        for bbox2=1:size(STAS{k, 1},1)
            box2 = STAS{k, 1}(bbox2,:);
            iou{bbox2,bbox} = bboxOverlapRatio(box(bbox,:),box2);
        end
    end
    for bbox2=1:size(STAS{k, 1},1)
        box2 = STAS{k, 1}(bbox2,:);
        if sum(cell2mat(iou(bbox2,:)))>0.5
            nobox=[nobox;box2];
            nostas{k,1} = nobox;
        end
    end
end
stas = gTruth.LabelData  ;


nostas = table(nostas);
labeldata = [stas,nostas];
gTruth1 = groundTruth(dataSource,labelDefs,nostas)