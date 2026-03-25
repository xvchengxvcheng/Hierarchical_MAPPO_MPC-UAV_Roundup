import numpy as np
from rtree import index
from matplotlib.pyplot import Rectangle
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D

class Map:
  def __init__(self, obstacle_list, bounds, path_resolution = 0.5, dim = 3):
    '''initialise map with given properties'''
    self.dim = dim
    self.idx = self.get_tree(obstacle_list, dim)
    self.len = len(obstacle_list)
    self.path_res = path_resolution
    self.obstacles = obstacle_list
    self.bounds = bounds

  @staticmethod
  def get_tree(obstacle_list, dim):
    '''initialise map with given obstacle_list'''
    p = index.Property()
    p.dimension = dim
    ls = [(i,(*obj,),None) for i, obj in enumerate(obstacle_list)]
    return index.Index(ls, properties=p)

  def add(self, obstacle):
    '''add new obstacle to the list'''
    self.idx.insert(self.len, obstacle)
    self.obstacles.append(obstacle)
    self.len += 1


  def inbounds(self,p):
    '''Check if p lies inside map bounds'''
    lower,upper = self.bounds
    return (lower <= p).all() and (p <= upper).all()

  def plotobs(self,ax,scale = 1):
    '''plot all obstacles'''
    obstacles = scale*np.array(self.obstacles)
    if self.dim == 2:
        for box in obstacles:
            l = box[2] - box[0]
            w = box[3] - box[1]
            box_plt = Rectangle((box[0], box[1]),l,w,color='k',zorder = 1)
            ax.add_patch(box_plt)
    elif self.dim == 3:
        for box in obstacles:
            X, Y, Z = cuboid_data(box)
            # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color= (0.10, 0.55, 0.95, 0.30), zorder = 1)
            ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color=(0.35, 0.65, 0.95, 0.18), zorder = 1)
            # color = (0.1, 0.15, 0.3, 0.2) (0.35, 0.65, 0.95, 0.18)
    else: print('can not plot for given dimensions')

  def obstacles_in_cube(self, p, r):
      """
      返回所有和 [p-r, p+r] 这个轴对齐立方体 (边长 2r) 有交叠的障碍物 ID。
      p: np.array([x,y]) 或 [x,y,z]
      r: 半径 (float)
      """
      p = np.asarray(p, dtype=float)

      # 1. 构造查询包围盒 (AABB)，拿它去问 R-tree
      if self.dim == 2:
          query_bb = (p[0] - r, p[1] - r,
                      p[0] + r, p[1] + r)
      elif self.dim == 3:
          query_bb = (p[0] - r, p[1] - r, p[2] - r,
                      p[0] + r, p[1] + r, p[2] + r)
      else:
          raise ValueError("Unsupported dim")

      # 2. 粗筛：哪些障碍物的盒子跟这个查询盒子有交集？
      cube_ids = list(self.idx.intersection(query_bb))
      cube_ids_set = set(cube_ids)

      # 2. 用这些候选障碍再建一个局部R-tree
      #    这样后面每条射线都可以只问这个小树，而不是整个 self.idx
      if len(cube_ids) == 0:
          local_idx = None
      else:
          prop = index.Property()
          prop.dimension = self.dim
          sub_items = [
              (oid, (*self.obstacles[oid],), None)
              for oid in cube_ids
          ]
          local_idx = index.Index(sub_items, properties=prop)

      return cube_ids_set, local_idx

  def raycast_segment(self, p0, p1, local_idx, return_all=False):
      """
      在三维空间中，查询从 p0 -> p1 的线段(射线段)与障碍物的碰撞情况。

      参数:
          p0, p1: np.array([x,y,z])
          return_all:
              False -> 只返回距离p0最近的撞击 (最早的交点)
              True  -> 返回所有相交障碍及其交点，按距离排序

      返回:
          如果 return_all == False:
              hit_info or None
              其中 hit_info 是字典:
              {
                  "t": 线段参数t in [0,1]，越小越靠近p0,
                  "point": 交点 np.array([x,y,z]),
              }
              没命中则返回 None

          如果 return_all == True:
              命中列表 hits_sorted (可能是空list)，列表中每个元素跟上面hit_info结构一样
      """
      p0 = np.asarray(p0, dtype=float)
      p1 = np.asarray(p1, dtype=float)

      if local_idx is None:
          return [] if return_all else None

      # 1. 用线段的AABB去R-tree做粗筛
      seg_bb = (
          min(p0[0], p1[0]),
          min(p0[1], p1[1]),
          min(p0[2], p1[2]),
          max(p0[0], p1[0]),
          max(p0[1], p1[1]),
          max(p0[2], p1[2]),
      )

      cand_ids_line = list(local_idx.intersection(seg_bb))  # 可能被线段碰到的障碍 index 列表

      hits = []

      # 2. 对候选障碍做精确求交
      for oid in cand_ids_line:
          box = self.obstacles[oid]
          hit, t_entry, pt_entry = segment_aabb_intersection(p0, p1, box)
          if hit:
              hits.append({
                  "t": t_entry,
                  "point": pt_entry
              })

      if not hits:
          return [] if return_all else None

      # 3. 按距离p0的先后顺序排序 (t越小越早撞到)
      hits_sorted = sorted(hits, key=lambda h: h["t"])

      if return_all:
          return hits_sorted
      else:
          return hits_sorted[0]["point"]


#to plot obstacle surfaces
def cuboid_data(box):
    l = box[3] - box[0]
    w = box[4] - box[1]
    h = box[5] - box[2]
    x = [[0, l, l, 0, 0],
         [0, l, l, 0, 0],
         [0, l, l, 0, 0],
         [0, l, l, 0, 0]]
    y = [[0, 0, w, w, 0],
         [0, 0, w, w, 0],
         [0, 0, 0, 0, 0],
         [w, w, w, w, w]]
    z = [[0, 0, 0, 0, 0],
         [h, h, h, h, h],
         [0, 0, h, h, 0],
         [0, 0, h, h, 0]]
    return box[0] + np.array(x), box[1] + np.array(y), box[2] + np.array(z)

# Generate random obstacle parameters
def random_grid_3D(bounds, density, height, size):
  if size >= 100:
      x_size = size
      y_size = size
      z_size = size
  else:
      x_size = 100
      y_size = 100
      z_size = 100

  # Generate random grid with discrete 0/1 altitude using normal distribtion
  mean_E = 0
  sigma = 1
  k_sigma = density
  E = np.random.normal(mean_E, sigma, size=(x_size, y_size))
  h = height

  # Set the decision threshold
  sigma_obstacle = k_sigma * sigma
  E = E > sigma_obstacle
  E = E.astype(float)

  # Generate random altitude to blocks
  h_min = 10 # minimal obstacles altitude
  E_temp = E
  for i in range(x_size):
      for j in range(y_size):
          #k = range(i - 1 - round(np.random.beta(0.5, 0.5)), i + 1 + round(np.random.beta(0.5, 0.5)), 1)
          #l = range(j - 1 - round(np.random.beta(0.5, 0.5)), j + 1 + round(np.random.beta(0.5, 0.5)), 1)

          if E_temp[j,i]==1:
              hh = round(np.random.normal(h, 0.7*h))
              if hh < h_min:
                  hh = h_min
              elif hh > z_size:
                  hh = z_size
              E[j,i] = hh
  return E

# seed = 42, 10, 20, 25
def generate_map(bounds, density, height, start, goal, size, path_resolution = 0.1, seed_num=42):
  np.random.seed(seed_num)
  # Create the obstacles on the map
  obstacles_ = random_grid_3D(bounds,density,height,size)
  obstacles = []

  if size >= 100:
      x_size = size
      y_size = size
  else:
      x_size = 100
      y_size = 100
  scale_ = bounds[1]/x_size

  for i in range(x_size):
    for j in range(y_size):
      if obstacles_[i,j] > 0:
        ss = round(np.random.normal(3, 1)) # Define the parameter to randomize the obstacle size
        obstacles.append([i * scale_, j * scale_, 0, (i+ss)* scale_, (j+ss)* scale_, obstacles_[i,j] * scale_])
        
  # create map with selected obstacles
  obstacles = start_goal_mapcheck(start,goal,obstacles)
  mapobs = Map(obstacles, bounds, dim = 3, path_resolution=path_resolution)
  print('Generate %d obstacles on the random map.'%len(obstacles))
  return mapobs, obstacles

# Check if the start point and the goal point on the map  
def start_goal_mapcheck(starts, goal, obstacles, margin=0.5):
    obs_pop = set()

    for j in range(starts.shape[0]):
        start = starts[j]

        for i, box in enumerate(obstacles):
            xmin, ymin, zmin, xmax, ymax, zmax = box

            # 膨胀障碍物边界：四周各加 margin
            xmin_m = xmin - margin
            ymin_m = ymin - margin
            zmin_m = zmin - margin
            xmax_m = xmax + margin
            ymax_m = ymax + margin
            zmax_m = zmax + margin

            start_inside = (
                xmin_m <= start[0] <= xmax_m and
                ymin_m <= start[1] <= ymax_m and
                zmin_m <= start[2] <= zmax_m
            )

            goal_inside = (
                xmin_m <= goal[0] <= xmax_m and
                ymin_m <= goal[1] <= ymax_m and
                zmin_m <= goal[2] <= zmax_m
            )

            if start_inside or goal_inside:
                # 避免同一障碍物被多次起点命中时重复打印
                if i not in obs_pop:
                    print("The start or goal collides with an obstacle (with margin)!")
                obs_pop.add(i)

    for idx in sorted(obs_pop, reverse=True):
        obstacles.pop(idx)

    return obstacles

def segment_aabb_intersection(p0, p1, box):
    """
    p0, p1: np.array([x,y,z])
    box: [xmin, ymin, zmin, xmax, ymax, zmax]
    返回:
        hit (bool): 是否相交
        t_entry (float): 线段参数t，0表示p0，1表示p1；只有在hit=True时有意义
        entry_pt (np.array): 第一个撞到盒子的交点坐标 (p0->p1方向上最近的点)
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)

    dir_vec = p1 - p0  # 方向向量
    box_min = np.array([box[0], box[1], box[2]], dtype=float)
    box_max = np.array([box[3], box[4], box[5]], dtype=float)

    tmin = 0.0  # 当前已知“最晚的进入时间”
    tmax = 1.0  # 当前已知“最早的离开时间”

    for axis in range(3):  # x=0,y=1,z=2
        if abs(dir_vec[axis]) < 1e-12:
            # 线段在这个轴上基本是平行的（方向量≈0）
            # 如果起点在盒子该轴范围外 -> 永远进不去
            if p0[axis] < box_min[axis] or p0[axis] > box_max[axis]:
                return False, None, None
            # 否则这个轴不收缩t范围（线段在这个轴上本来就“贴着/在里面”）
        else:
            # 计算线段什么时候穿过 box_min 和 box_max 这两道平面
            t1 = (box_min[axis] - p0[axis]) / dir_vec[axis]
            t2 = (box_max[axis] - p0[axis]) / dir_vec[axis]

            # 保证 t1 <= t2
            if t1 > t2:
                t1, t2 = t2, t1

            # 和总的进入/离开时间做交集
            if t1 > tmin:
                tmin = t1
            if t2 < tmax:
                tmax = t2

            # 如果时间区间已经空了，就说明没有交集
            if tmin > tmax:
                return False, None, None

    # 走到这里说明线段和盒子有交叠区间 [tmin, tmax]
    # 我们关心的是“从p0往p1方向首先撞到的位置”，也就是 t_entry = tmin
    # 但必须保证这个撞击点真的在段内: t in [0,1]
    if tmin < 0 or tmin > 1:
        # 有可能线段起点一开始就在盒子内部，这时tmin<0但tmax>=0
        # 如果p0已在盒子里，可按“碰撞从起点开始”处理
        inside = np.all(p0 >= box_min) and np.all(p0 <= box_max)
        if inside:
            t_entry = 0.0
        else:
            return False, None, None
    else:
        t_entry = tmin

    entry_pt = p0 + t_entry * dir_vec
    return True, float(t_entry), entry_pt


def main():
  # limits on map dimensions
  bounds = np.array([0,100])
  size = 250

  # Define the density value of the map
  density = 2.5

  # Define the height parameter of the obstacles on the map
  height = 50

  # Define the start point and goal point
  start = np.array([[0,0,5]])
  goal = np.array([65,65,5])

  # create map with selected obstacles
  mapobs,obstacles = generate_map(bounds, density, height, start, goal, size)

  p0 = np.array([40.0, 40.0, 20.0])
  p1_list = [
      np.array([50.0, 40.0, 20.0]),
      np.array([40.0, 50.0, 15.0]),
      np.array([46.0, 46.0, 20.0]),
      np.array([34.0, 34.0, 15.0]),
      # ... 你可以加更多方向/终点
  ]

  colls = []
  cube_ids, local_idx = mapobs.obstacles_in_cube(p0, r=20.0)
  for p1 in p1_list:
      hit_info = mapobs.raycast_segment(p0, p1, local_idx, return_all=False)

      if hit_info is None:
          print("p0 ->", p1, "：没有撞到障碍")
      else:
          colls.append(hit_info)

  # Visualize the obstacle map 
  fig = plt.figure(figsize=(13, 13))
  ax = fig.add_subplot(111, projection='3d')  # ✅ 推荐方式
  ax.set_box_aspect((1, 1, 1))  # 立方体比例
  ax.set_xlim(bounds[0], bounds[1])
  ax.set_ylim(bounds[0], bounds[1])
  ax.set_zlim(bounds[0], bounds[1])
  mapobs.plotobs(ax, scale=1)

  # ax.scatter(p0[0], p0[1], p0[2], s=60, c='g', marker='o')  # 起点 绿色
  # for i in range(len(colls)):
  #     ax.plot(
  #         [p0[0], colls[i][0]],  # x 坐标序列
  #         [p0[1], colls[i][1]],  # y 坐标序列
  #         [p0[2], colls[i][2]],  # z 坐标序列
  #         linewidth=3,  # 线粗一点
  #         color='r'  # 红色线（你可以改成别的）
  #     )  # <<< 新增：射线段
  #
  #     # 3. （可选）把起点和撞击点画成散点，方便看
  #     ax.scatter(colls[i][0], colls[i][1], colls[i][2], s=60, c='r', marker='^')



  ax.zaxis.set_visible(False)
  ax.xaxis.pane.set_visible(False)
  ax.yaxis.pane.set_visible(False)
  ax.grid(False)

  ax.view_init(elev=60, azim=45)
  # plt.tight_layout()
  fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95)

  plt.show()

'''Call the main function'''
if __name__ == "__main__":
  main()

  