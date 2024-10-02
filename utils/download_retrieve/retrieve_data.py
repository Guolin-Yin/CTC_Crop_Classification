import os
import numpy as np
from shapely.geometry import shape, box
import scipy.io
import json
import csv
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
from datetime import datetime
import concurrent.futures
import logging
from pathlib import Path

# 1. 解析 JSON 文件中的 Polygon 数据
def load_polygons_from_json(json_path):
    with open(json_path) as f:
        data = json.load(f)

    polygons = []
    for feature in data['features']:
        geometry = shape(feature['geometry'])
        properties = feature['properties']
        sample_name = f"{feature['id']}_{properties['VARIEDADE']}"
        polygons.append({'geometry': geometry, 'SampleName': sample_name, 'properties': properties})

    return polygons

# 2. 从 .mat 文件中按需加载 tile 信息以及波段表头
# 处理瓦片的 size 数据，提取实际的 height 和 width
def load_tiles_and_band_names_from_mat(path):
    data = scipy.io.loadmat(path)
    tiles = {}
    band_names = []

    for key in data.keys():
        if key.startswith('tile_'):
            try:
                min_x, max_x, min_y, max_y = data[key][0][0][0][0][0]
                size = data[key][0][0][1]

                # 处理嵌套的二维数组，提取实际的 (height, width)
                height, width = size[0][0], size[0][1]

                img = np.transpose(data[key][0][0][2], (0, 2, 1))  # 将图像数据转置为 (bands, height, width)

                # 检查 img 是否是三维数据
                if img.ndim != 3:
                    print(f"Warning: Unexpected image dimensions in {key}: {img.shape}")
                    continue

                tiles[key] = {
                    'bbox': (min_x, max_x, min_y, max_y),
                    'size': (height, width),
                    'img': img
                }
            except IndexError as e:
                print(f"Error processing tile {key} in {path}: {e}")
                continue

    if 'band_names' in data:
        band_names = data['band_names'].flatten().tolist()
    else:
        band_names = [f'Band_{i+1}' for i in range(14)]

    return tiles, band_names

# 3. 创建瓦片的仿射变换矩阵
def create_transform(min_x, max_x, min_y, max_y, width, height):
    return from_bounds(min_x, min_y, max_x, max_y, width, height)

# 4. 栅格化多边形生成掩码
def polygon_to_mask(polygon, tile_bbox, tile_size):
    min_x, max_x, min_y, max_y = tile_bbox

    # 处理意外的尺寸
    try:
        height, width = tile_size[0], tile_size[1]
    except IndexError:
        print(f"Invalid tile size: {tile_size}")
        return None

    transform = create_transform(min_x, max_x, min_y, max_y, width, height)

    try:
        mask = geometry_mask([polygon], transform=transform, invert=True, out_shape=(height, width))
    except Exception as e:
        print(f"Error creating mask: {e}")
        return None

    return mask

# 5. 预计算样本的瓦片相交关系与掩码
def precompute_tile_masks_for_sample(polygon, tiles):
    tile_masks = {}
    for tile_name, tile_info in tiles.items():
        min_x, max_x, min_y, max_y = tile_info['bbox']
        if polygon.intersects(box(min_x, min_y, max_x, max_y)):
            mask = polygon_to_mask(polygon, tile_info['bbox'], tile_info['size'])
            if mask is not None:
                tile_masks[tile_name] = mask
    return tile_masks

# 6. 从指定时间范围内加载 .mat 文件
def load_mat_files_in_time_range(folder_path, start_date, end_date):
    mat_files = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.mat'):
            try:
                if '_to_' in file_name:
                    date_part = file_name.split('_to_')
                    file_start_date = date_part[0].split('_')[-1]  # 获取开始日期
                    file_end_date = date_part[1].replace('.mat', '').strip()  # 获取结束日期

                    file_start_date = datetime.strptime(file_start_date, '%Y-%m-%d')
                    file_end_date = datetime.strptime(file_end_date, '%Y-%m-%d')

                    if (file_start_date <= end_date) and (file_end_date >= start_date):
                        mat_files.append(os.path.join(folder_path, file_name))
            except ValueError as e:
                print(f"Error parsing dates from file name: {file_name}. Error: {e}")
            except Exception as e:
                print(f"Unexpected error with file {file_name}: {e}")

    return mat_files

# 7. 批量掩码应用
def apply_mask_and_extract(tile_info, mask):
    img = tile_info['img']
    height, width = tile_info['size'][0], tile_info['size'][1]
    lon_grid = np.linspace(tile_info['bbox'][0], tile_info['bbox'][1], width)
    lat_grid = np.linspace(tile_info['bbox'][3], tile_info['bbox'][2], height)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    # 使用掩码直接筛选出所有的值
    return lon_mesh[mask], lat_mesh[mask], img[:, mask].T

# 8. 处理单个 .mat 文件的样本
def process_single_file(polygon_data, mat_file, tile_masks_cache):
    try:
        tiles, _ = load_tiles_and_band_names_from_mat(mat_file)
    except Exception as e:
        print(f"Error loading tiles from {mat_file}: {e}")
        return []

    rows = []
    sample_name = polygon_data['SampleName']
    polygon = polygon_data['geometry']

    source_file_name = os.path.basename(mat_file).replace('Alldata_', '').replace('.mat', '')

    # 如果没有缓存掩码，计算一次
    if tile_masks_cache is None:
        tile_masks_cache = precompute_tile_masks_for_sample(polygon, tiles)

    for tile_name, tile_info in tiles.items():
        if tile_name in tile_masks_cache:
            mask = tile_masks_cache[tile_name]
            if mask is not None:
                lon_values, lat_values, band_values = apply_mask_and_extract(tile_info, mask)
                for lat, lon, bands in zip(lat_values, lon_values, band_values):
                    rows.append([sample_name, lat, lon, source_file_name] + bands.tolist())

    return rows, tile_masks_cache

# 9. 优化后的按样本并行保存数据
def find_and_save_polygon_data_parallel(polygons, folder_path, save_dir, band_names, start_date, end_date):
    mat_files = load_mat_files_in_time_range(folder_path, start_date, end_date)
    counter_skip = 0
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for polygon_data in polygons:
            sample_name = polygon_data['SampleName']
            total_points_saved = 0
            output_csv = save_dir / f"{sample_name}_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}.csv"
            # check if it already exists
            if output_csv.exists():
                counter_skip += 1
                continue
            logging.info(f"Skipping {counter_skip} plolygons.\n")
            # 为每个样本创建一个单独的CSV文件
            
            rows = []
            headers = ['SampleName', 'Latitude', 'Longitude', 'SourceFile'] + band_names
            rows.append(headers)

            result_rows, tile_masks_cache = process_single_file(polygon_data, mat_files[0], None)
            rows.extend(result_rows)
            total_points_saved += len(result_rows)

            # remove the processed file

            # Process the rest of the files
            future_to_file = {
                executor.submit(process_single_file, polygon_data, mat_file, tile_masks_cache): mat_file
                for mat_file in mat_files[1:]
            }

            for future in concurrent.futures.as_completed(future_to_file):
                mat_file = future_to_file[future]
                try:
                    result_rows,_ = future.result()
                    rows.extend(result_rows)
                    total_points_saved += len(result_rows)
                except Exception as exc:
                    print(f"{mat_file} generated an exception: {exc}")
                    logging.error(f"Error processing {mat_file}: {exc}")

            # 所有数据处理完后再一次性写入 CSV 文件
            
            logging.info(f"Polygon {sample_name} processed with {total_points_saved} points saved across {len(mat_files)} files.\n")
            with open(output_csv, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(rows)

if __name__ == '__main__':
    logging.basicConfig(filename='app_2023.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # 使用示例
    json_path = './data/Shape_labels/filtered_output_data_geo.json'

    # 1. 加载 Polygon 数据
    polygons = load_polygons_from_json(json_path)

    start_date = datetime(2023, 4, 1)
    end_date = datetime(2024, 1, 5)

    folder_path = Path('./data/Raw_data-2023')
    save_dir = Path('./data/processed_2023')


    save_dir.mkdir(parents=True, exist_ok=True)

    log_file = Path('./log.txt')
    for week_data in list(folder_path.glob('*.mat')):
        print(f"Processing {week_data}")
        tiles, band_names = load_tiles_and_band_names_from_mat(week_data)
        print(f"Data saved in individual CSV files and log saved to {log_file}")
        break
    find_and_save_polygon_data_parallel(polygons, folder_path, save_dir, band_names, start_date, end_date)

