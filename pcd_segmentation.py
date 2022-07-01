from fileinput import filename
from pclpy import pcl
import math
import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm

# 预处理后的pcd点云
# 区域生长 --> clusters
# for each cluster：
    # 提取cluster索引的point cloud
    # 先计算协方差矩阵pca的特征向量，看类内差异性，分成不同类
    # if cofe > . : ransec 一次，分成两个cluster_
    # 对于每个cluster，计算ransec的法向量，得到y轴。(0，0，1)为x轴。叉乘得到z轴向量
    # 点云局部坐标系转换及投影

# PCD io
RawPCDName = '103-3rd-floor_voxel_section_remove.pcd'
PCD_FilePath = './output_data/'
PCD_Trans_FilePath = './output_data_trans/'

# region_grow parameters
RadiusSearch = 0.35
MaxClusterSize = 500000
MinClusterSize = 500
NumberOfNeighbours = 30
SmoothnessThreshold = 5 / 180 * math.pi                          
CurvatureThreshold = 10
ResidualThreshold = 1
RegionGrowVisual=True  #可视化region_grow结果，仅限pclpy=0.11.0及以下使用，0.12.0需要视为False且另外使用Open3d可视化

GridSize = 0.2   

# 协方差矩阵主轴方向计算s
def SVD_PCD(pcd):
    centroid = pcd.xyz.mean(axis=0)
    A = (pcd.xyz-centroid).T
    u,s,vh = np.linalg.svd(A,full_matrices=False, compute_uv=True)

    pcd_direction = (u[:,-1])
    return pcd_direction

# ransac输出inliers,outliner和内点平面法向量coeff
def Ransac_PCD(pcd):
    inliers = pcl.vectors.Int()
    model_p = pcl.sample_consensus.SampleConsensusModelPlane.PointXYZ(pcd)
    ransac = pcl.sample_consensus.RandomSampleConsensus.PointXYZ(model_p)
    ransac.setDistanceThreshold(0.5)
    ransac.computeModel()             # 进行拟合
    ransac.getInliers(inliers)        # 获取内点索引
    coeffs = pcl.vectors.VectorXf()   # 存储平面系数的vector(pcl.vectors.VectorXf()为Eigen类型对应的python表示)
    ransac.getModelCoefficients(coeffs)  # 获取系数
    coeff = np.array(coeffs)          # 转换为numpy所支持的格式

    inliers_cloud = pcl.PointCloud.PointXYZ()
    outliers_cloud = pcl.PointCloud.PointXYZ()
    extract = pcl.filters.ExtractIndices.PointXYZ()
    extract.setInputCloud(pcd)
    extract.setIndices(inliers)
    extract.setNegative(False)
    extract.filter(inliers_cloud)
    extract.setNegative(True)
    extract.filter(outliers_cloud)

    return coeff,inliers_cloud,outliers_cloud

# ransac平面与点云主方向夹角
def Calcu_Theta(coeff,s):
    cos_theta = np.sum(coeff[:3] * s)/((np.sqrt(np.sum(coeff[:3]**2)))*(np.sqrt(np.sum(s**2))))
    theta = np.degrees(np.arccos(cos_theta))
    # print(cos_theta)
    # print(theta)
    return theta

def Split_Cluster(it, pcd, pcd_split):
    pcd_direction = SVD_PCD(pcd)
    print("主轴方向为：", pcd_direction)

    coeff, inliers_cloud,outliers_cloud = Ransac_PCD(pcd)
    theta = Calcu_Theta(coeff,pcd_direction)
    print("夹角为：",theta)

    itnum = 1
    file_name = "Cluster_%06d" % it
    # theta 为 [0,180]
    break_bool = False
    while theta > 20 and theta < 160 and np.size(outliers_cloud.xyz,0) >= 300:
        pcd_split = True
        pcl.io.savePCDFile(PCD_FilePath + file_name+"_2"*(itnum-1) + '_1.pcd', inliers_cloud)

        # 如果theta范围为[20,160]，继续ransac
        _ = outliers_cloud
        pcd_direction = SVD_PCD(outliers_cloud)
        coeff, inliers_cloud, outliers_cloud = Ransac_PCD(outliers_cloud)
        theta = Calcu_Theta(coeff,pcd_direction)
        if np.size(outliers_cloud.xyz,0) < 300 : 
            outliers_cloud = _
            pcl.io.savePCDFile(PCD_FilePath + file_name+"_2"*(itnum)+'.pcd', outliers_cloud)
            break_bool = True
            break
        itnum+=1
    
    if break_bool: pass
    elif pcd_split==True and break_bool==False: pcl.io.savePCDFile(file_name+"_2"*itnum+'.pcd', outliers_cloud)

    return pcd_split, theta


def Region_Grow(pcd):
    
    normals_estimation = pcl.features.NormalEstimationOMP.PointXYZ_Normal()
    normals_estimation.setInputCloud(pcd)
    normals = pcl.PointCloud.Normal()
    normals_estimation.setRadiusSearch(RadiusSearch)
    normals_estimation.compute(normals)

    # 创建区域生长分割对象
    rg = pcl.segmentation.RegionGrowing.PointXYZ_Normal()   
    # 初始化点云、法向量
    rg.setInputCloud(pcd)                                       
    rg.setInputNormals(normals)                                 
    # 最大、最小cluster点数量
    rg.setMaxClusterSize(MaxClusterSize)                              
    rg.setMinClusterSize(MinClusterSize)                                   
    # knn邻域点数量
    rg.setNumberOfNeighbours(NumberOfNeighbours)  
    # 平滑度（法向量）阈值、表面曲率阈值、残差阈值                              
    rg.setSmoothnessThreshold(SmoothnessThreshold)               
    rg.setCurvatureThreshold(CurvatureThreshold)                                 
    rg.setResidualThreshold(ResidualThreshold)                                 
    # 获取分割索引
    clusters = pcl.vectors.PointIndices()                      
    rg.extract(clusters)
    
    if RegionGrowVisual:
        colored_cloud = pcl.PointCloud.PointXYZRGBA()
        colored_cloud = rg.getColoredCloud()
        viewer = pcl.visualization.PCLVisualizer("RegionGrowing")
        viewer.setBackgroundColor(0, 0, 0)
        viewer.addPointCloud(colored_cloud, "RegionGrowing cloud")
        viewer.setPointCloudRenderingProperties(0, 1, "RegionGrowing cloud")
        while not viewer.wasStopped():
            viewer.spinOnce(10) 

    return clusters

def Extract_Indices(pcd, indices):
    pcd_extract = pcl.PointCloud.PointXYZ()
    extract = pcl.filters.ExtractIndices.PointXYZ()
    extract.setInputCloud(pcd)
    extract.setIndices(indices)
    extract.setNegative(False)  # 设置为false表示提取对应索引之内的点
    extract.filter(pcd_extract)
    return pcd_extract

def TransToOtho(coeff, pcdInput, normalVector=None, shallUseGroundPlaneNormal=0):
    transMat = np.eye(4)
    pointMat = pcdInput.xyz
    if not shallUseGroundPlaneNormal:
        zAxis = np.array([0,0,1])
    else:
        zAxis = normalVector  # TODO: logic of using ground plane normal.
    # if zAxis == None:
    #     return None
    transMat[0, :3] = zAxis
    transMat[2, :3] = coeff[:3]
    transMat[1, :3] = np.cross(transMat[0, :3], transMat[2, :3])
    transMat[:3, 3] = pointMat.mean(axis=0)

    xyzTrans = np.dot(transMat[:3, :3], pointMat.T).T + transMat[:3, 3].T
    return xyzTrans, transMat

def ProjectPointCloudToImage(filepath, filename, points, GridSize):
    pts2d = points[:, :2]
    boarderMin = np.min(pts2d, axis=0)
    boarderMax = np.max(pts2d, axis=0)

    x, y = np.mgrid[boarderMin[0]:boarderMax[0]:GridSize, boarderMin[1]:boarderMax[1]:GridSize]
    rows = np.floor((pts2d[:,0]-boarderMin[0])/GridSize).astype(np.int)
    cols = np.floor((pts2d[:,1]-boarderMin[1])/GridSize).astype(np.int)
    z = np.empty(x.shape)
    z.fill(np.nan)
    z[(rows, cols )] = 1

    SaveImage(z, boarderMin, boarderMax, filepath, filename)
    return z, boarderMin, boarderMax


def Save_PCD(FilePath, it, pcd):
    file_name = FilePath + "Cluster_%06d.pcd" % it
    # pcl.io.savePCDFile(file_name, pcd)
    pcl.io.savePCDFileASCII(file_name, pcd) 

def Save_Cluster(FilePath, pcd, clusters):    
    it = 0
    for c in clusters:
        # 定义分类点的索引 inliers
        inliers = pcl.PointIndices()
        # 计算分类点的索引 inliers
        for i in c.indices:
            inliers.indices.append(i)    
        # 根据分类点的索引inliers提取cloud_inliers
        cloud_inliers = Extract_Indices(pcd, inliers)

        pcd_split = False
        pcd_split,_ = Split_Cluster(it,cloud_inliers,pcd_split)
        if not pcd_split: Save_PCD(FilePath, it, cloud_inliers)
        
        it += 1

def SaveImage(z, boarderMin, boarderMax, filepath, filename):
    zx,zy = np.where(z==1)

    plt.figure(figsize=(boarderMax[0]-boarderMin[0],boarderMax[1]-boarderMin[1]),dpi=50)
    plt.scatter(zx,zy,c='b',marker='s',linewidths=10)
    plt.xlim([boarderMin[0], boarderMax[0]])
    plt.ylim([boarderMin[1], boarderMax[1]])
    plt.axis('equal')
    plt.grid()
    plt.savefig(filepath + filename + '_imagegrid.png')
    plt.close() # 防止爆内存

# 创建文件夹    
def mkdir(path):
    folder = os.path.exists(path)
    if folder: 
        shutil.rmtree(path)                  #判断是否存在文件夹，如果不存在则创建为文件夹
    os.makedirs(path)                        #makedirs 创建文件时如果路径不存在会创建这个路径
	
        
def main():
    # 创建文件夹
    mkdir(PCD_FilePath)
    mkdir(PCD_Trans_FilePath)
    # pcd input
    pcd = pcl.PointCloud.PointXYZ()
    pcl.io.loadPCDFile(RawPCDName, pcd)

    # region grow
    print('区域生长中...')
    clusters = Region_Grow(pcd)
    print('区域生长完成！')
    print('切分类别中...')
    Save_Cluster(PCD_FilePath, pcd, clusters)
    print('切分类别完成！')

    # 局部坐标系投影
    print('局部坐标系投影中...')
    PCD_Files= os.listdir(PCD_FilePath)
    boarder_sum = np.zeros(shape=(0,4))
    for file in tqdm(PCD_Files):
        pcd = pcl.PointCloud.PointXYZ()
        pcl.io.loadPCDFile(PCD_FilePath + file, pcd)
        coeff, _, _ = Ransac_PCD(pcd)
        pcd_Trans, transMat = TransToOtho(coeff, pcd, normalVector=None, shallUseGroundPlaneNormal=0)
        
        pcd1 = pcl.PointCloud.PointXYZ(pcd_Trans)
        index = file.rfind('.')
        filename = file[:index]
        np.savetxt(PCD_Trans_FilePath + filename + '_transMat.csv',transMat, delimiter=',') # 投影平面的转换矩阵
        pcl.io.savePCDFile(PCD_Trans_FilePath + filename + '_proj.pcd', pcd1) # 投影平面后的网格可视化图片

        z, boarderMin, boarderMax = ProjectPointCloudToImage(PCD_Trans_FilePath, filename, pcd_Trans, GridSize)
        np.savetxt(PCD_Trans_FilePath + filename + '_imagegrid.csv', z, delimiter=',') # 投影平面后的网格矩阵

        boarder = np.hstack((boarderMin, boarderMax)).reshape(1,-1)
        boarder_sum = np.concatenate((boarder_sum, boarder), axis=0)

    np.savetxt(PCD_Trans_FilePath + 'Clusters_boarder_sum.csv', boarder_sum, delimiter=',') # 边界点统计

if __name__ == "__main__":
    main()
