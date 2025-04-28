import numpy as np
import torch

def distance_fun(xyz, centroid):
    # return torch.cosine_similarity(xyz, centroid, -1)
    return torch.sum((xyz - centroid) ** 2, -1)

def farthest_point_sample_with_cuda(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    # 创建大小为(B, npoint)的零张量，用于存储采样点的索引
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    # 创建大小为(B, N)的张量，并初始化为一个较大的值，用于存储每个点到采样点的距离
    distance = torch.ones(B, N).to(device) * 1e10
    print("distance:", distance)
    # 从点云中随机选择一个点作为初始采样点
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    selected_embedding = {}

    for i in range(npoint):#循环采样npoint个点
        # 将当前最远点的索引添加到centroids中
        centroids[:, i] = farthest
        # 获取上一个采样点的坐标，形状为(B, 1, 3)
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 1024)

        #欧式距离
        dist = distance_fun(xyz, centroid)
        # print("*******************dist:",dist)
        # return
        mask = dist < distance
        distance[mask] = dist[mask]
        # print("distance:", distance)
        # 选择距离最远的点作为下一个采样点
        farthest = torch.max(distance, -1)[1]

        selected_embedding[farthest.cpu().item()] = xyz[batch_indices, farthest, :].view(B, 1, 1024).squeeze().cpu().numpy()
        print(selected_embedding)
    return centroids, selected_embedding

# 大数据集分批采样
def fps_with_index(xyz, npoint, index):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    # 创建大小为(B, npoint)的零张量，用于存储采样点的索引
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    # 创建大小为(B, N)的张量，并初始化为一个较大的值，用于存储每个点到采样点的距离
    distance = torch.ones(B, N).to(device) * 1e10
    # 从点云中随机选择一个点作为初始采样点
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    selected_embedding = {}

    for i in range(npoint):#循环采样npoint个点
        # 将当前最远点的索引添加到centroids中
        centroids[:, i] = farthest
        # 获取上一个采样点的坐标，形状为(B, 1, 3)
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 1024)

        #欧式距离
        dist = distance_fun(xyz, centroid)

        mask = dist < distance
        distance[mask] = dist[mask]
        # 选择距离最远的点作为下一个采样点
        farthest = torch.max(distance, -1)[1]

        selected_embedding[farthest.cpu().item()+index] = xyz[batch_indices, farthest, :].view(B, 1, 1024).squeeze().cpu().numpy()
    
    return centroids, selected_embedding