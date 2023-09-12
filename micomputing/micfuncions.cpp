#include <iostream>
#include <queue>
#include <math.h>

using namespace std;

//struct Coord
//{
//  int n_dim;
//  float* coord;
//  Coord(Coord &x) {
//      n_dim = x.n_dim;
//      coord = x.coord;
//  }
//  Coord(int* x, float* spacing, int n_dim) {
//      for (int d = 0; d < n_dim; d++) {
//          *(coord + d) = (float)*(x + d) * *(spacing + d);
//      }
//  }
//};
//
//struct Node
//{
//  int n_dim;
//  int* pos;
//  int* anchor;
//  float* spacing;
//
//  float dis() const {
//      float cum_dis_sq = 0;
//      for (int d = 0; d < n_dim; d++) {
//          float dx = (float)(*(pos + d) - *(anchor + d)) * *(spacing + d);
//          cum_dis_sq += dx * dx;
//      }
//      return sqrtf(cum_dis_sq);
//  }
//
//  bool operator < (const Node &other) const {
//      return dis() < other.dis();
//  }
//};
//
//void pos2coords(long pos, int* coords_out, int n_dim, int* shape_without_batch) {
//  for (int d = n_dim - 1; d >= 0; d--) {
//      long unit = (long)shape_without_batch[d];
//      *(coords_out + d) = pos % unit;
//      pos /= unit;
//  }
//}
//
//long coords2pos(int* coords, int n_dim, int* shape_without_batch) {
//  long pos = 0;
//  for (int d = 0; d < n_dim; d++) {
//      pos *= (long)(*(shape_without_batch + d));
//      pos += (long)(*(coords + d));
//  }
//  return pos;
//}

extern "C" {

    void distance_map(int* mask_in, float* dismap_out, int n_dim_with_batch, int* shape_with_batch, float* spacing_without_batch) {
        /*
        Compute the distance map of the mask given in mask_in, with the outside distance being positive and inside being negative. 
        [Usage] distance_map(int* mask_in, float* dismap_out, int n_dim_with_batch, int* shape_with_batch, float* spacing_without_batch)
        [Arguments]
            mask_in (int[n_batch * n_data]): The mask data (0-1 images) for n_batch number of masks, each with n_data elements. 
            dismap_out (int*): The pointer of the output, allocated with 0s of the same size as the input. 
            n_dim_with_batch (int): The number of dimensions including the batch dimension for parallels. Thus 3D image would have
            n_dim_with_batch = 1 + 3 = 4.
            shape_with_batch (int[1 + n_dim]): The shape of the tensor, with the first size the number of parallels, aka batch size. 
            spacing_without_batch (float[n_dim): The spacing of the image space, with the assumption that all parallels share the same spacing. 
        [Example]
            If we compute the distance map for a tensor in shape (20, 30, 10), we should input an integer array of length 1 x 20 x 30 x 10
            = 6000 (with values 0 and 1), a float array pointer with allocated length 6000, an integer n_dim_with_batch of 4 (being 1+3), an
            integer array indicating shape: [1, 20, 30, 10] (of length 4, with the first element indicating the number of parallels, i.e. 
            n_batch), and a float array indicating the image spacing, being [1., 1., 1.] (of length 3) by default. 
        [Note]
            If a mask is not binary, all non-zero elements will be considered as masked areas.
            If a mask is blank, without any non-zeros element, the output would be all zeros.
        */
        int n_dim = n_dim_with_batch - 1;
        int n_batch = shape_with_batch[0];
        long n_data = 1;
        int max_size = 0;
        long* index_unit = new long[n_dim];
        for (int d = n_dim; d > 0; d--) {
            // Compute the number of pixels for each image, and record the index change when moving in the dimension by 1.
            int size = shape_with_batch[d];
            index_unit[d-1] = n_data;
            n_data *= (long) size;
            if (size > max_size) max_size = size;
        }
        float inf = 1.1 * sqrtf((float)n_dim) * (float)max_size;
        
        for (int b = 0; b < n_batch; b++) {
            // Loop in the batch dimension.
            long image_start = n_data * b; // Compute the starting index of the current image in the batch.

            long* anchors = new long[n_data];
            bool* visited = new bool[n_data];
            for (long i = 0; i < n_data; i++) {
                // initialize the anchors and dismap tables;
                anchors[i] = -1;
                long pos = image_start + i;
                if (mask_in[pos] != 0) dismap_out[pos] = -inf; // inside: find the biggest and -1;
                else dismap_out[pos] = inf; // outside: find the smallest and +1;
            }
            // Begin: Apply the Danielsson algorithm.
            // [1] The EDM algorithm is similar to the 8SSEDT in F. Leymarie, M. D. Levine, in: CVGIP Image Understanding, vol. 55 (1992), pp 84-94
            for (int order = 0; order < 4; order++) {
                // scan forward and backward twice;
                for (long i = 0; i < n_data; i++) visited[i] = false;
                for (long i = 0; i < n_data; i++) {
                    long j;
                    if (order % 2 != 0) j = n_data - i - 1; // scan backward for odd 'order';
                    else j = i;
                    long pos = image_start + j;
                    for (int r = 0; r < n_dim; r++) {
                        int size_r = shape_with_batch[r + 1];
                        int coord_r = (j / index_unit[r]) % size_r;
                        for (int dir = -1; dir <= 1; dir += 2) {
                            if (coord_r + dir < 0 || coord_r + dir >= size_r) continue;
                            long neighbor = pos + dir * index_unit[r];
                            if (!visited[neighbor - image_start]) continue;
                            long k = anchors[neighbor - image_start];
                            if (mask_in[pos] != 0 && mask_in[neighbor] == 0 || mask_in[pos] == 0 && mask_in[neighbor] != 0)
                                k = neighbor - image_start;
                            long x = j, y = k;
                            float neighbor_dis = 0.;
                            if (k == -1) continue;
                            else {
                                for (int d = 0; d < n_dim; d++) {
                                    float dx = spacing_without_batch[d] * (float)(x / index_unit[d] - y / index_unit[d]);
                                    neighbor_dis += dx * dx;
                                    x %= index_unit[d];
                                    y %= index_unit[d];
                                }
                                neighbor_dis = sqrt(neighbor_dis);
                            }
                            if (mask_in[pos] != 0) {
                                if (- neighbor_dis > dismap_out[pos]) {
                                    dismap_out[pos] = - neighbor_dis;
                                    anchors[j] = k;
                                }
                            }
                            else if (neighbor_dis < dismap_out[pos]) {
                                dismap_out[pos] = neighbor_dis;
                                anchors[j] = k;
                            }
                        }
                    }
                    visited[j] = true;
                }
            }
            // End distance asignment.
            
        }
        
        return;
    }

//      int n_dim = n_dim_with_batch - 1;
//      int* shape_without_batch = new int[n_dim];
//      int n_batch = shape_with_batch[0];
//      long n_data = 1;
//      int max_size = 0;
//      for (int d = 1; d <= n_dim; d++) {
//          // Compute the number of pixels for each image, and record the shape without batch.
//          int size = *(shape_with_batch + d);
//          *(shape_without_batch + d - 1) = size;
//          n_data *= (long)size;
//          if (size > max_size) max_size = size;
//      }
//      int inf = (int)(2. * sqrtf((float)n_dim) * (float)max_size);
//  
//      for (int b = 0; b < n_batch; b++) {
//          // Loop in the batch dimension.
//          long image_start = n_data * b;
//          // Compute the starting index of the current image in the batch.
//
//          long* anchors = new long[n_data];
//          bool* visited = new bool[n_data];
//          for (long i = 0; i < n_data; i++) {
//              anchors[i] = -inf;
//              long pos = image_start + i;
//              if (mask_in[pos] != 0) dismap_out[pos] = -inf; // inside: find the biggest and -1;
//              else dismap_out[pos] = inf; // outside: find the smallest and +1;
//          }
//          // --Begin: Apply the Danielsson algorithm.
//          // The EDM algorithm is similar to the 8SSEDT in F. Leymarie, M. D. Levine, in: CVGIP Image Understanding, vol. 55 (1992), pp 84-94
//          for (int order = 0; order < 4; order++) {
//              for (long i = 0; i < n_data; i++) visited[i] = false;
//              for (long i = 0; i < n_data; i++) {
//                  long j;
//                  if (order % 2 != 0) j = n_data - i - 1;
//                  else j = i;
//                  long pos = image_start + j;
//                  int* coords = new int[n_dim];
//                  pos2coords(j, coords, n_dim, shape_without_batch);
//                  for (int d = 0; d < n_dim; d++) {
//                      for (int dir = -1; dir <= 1; dir += 2) {
//                          int* neighbor = new int[n_dim];
//                          bool out_of_bound = false;
//                          for (int k = 0; k < n_dim; k++) {
//                              if (k == d) {
//                                  int new_coord_at_dim = coords[k] + dir;
//                                  if (new_coord_at_dim >= 0 && new_coord_at_dim < shape_without_batch[d])
//                                      neighbor[k] = new_coord_at_dim;
//                                  else out_of_bound = true;
//                              }
//                              else neighbor[k] = coords[k];
//                          }
//                          if (out_of_bound) continue;
//                          long neighbor_i = coords2pos(neighbor, n_dim, shape_without_batch);
//                          long neighbor_pos = neighbor_i + image_start;
//                          if (!visited[neighbor_i]) continue;
////                          float cur_dis = dismap_out[pos];
////                          float neighbor_dis = dismap_out[neighbor_pos];
//                          float neighbor_dis = 0.;
//                          if (mask_in[pos] != 0 && mask_in[neighbor_pos] == 0 || mask_in[pos] == 0 && mask_in[neighbor_pos] != 0) {
//                              anchors[j] = neighbor_i;
//                              if (mask_in[pos] != 0) dismap_out[pos] = -1;
//                              else dismap_out[pos] = 1;
//                          }
//                          int* neighbor_anchor = new int[n_dim];
//                          pos2coords(anchors[neighbor_i], neighbor_anchor, n_dim, shape_without_batch);
//                          for (int k = 0; k < n_dim; k++) {
//                              float dx = spacing_without_batch[k] * (float)(coords[k] - neighbor_anchor[k]);
//                              neighbor_dis += dx * dx;
//                          }
//                          neighbor_dis = sqrt(neighbor_dis);
//                          if (mask_in[pos] != 0) {
//                              if (- neighbor_dis > dismap_out[pos]) {
//                                  dismap_out[pos] = - neighbor_dis;
//                                  anchors[j] = anchors[neighbor_i];
//                              }
//                          }
//                          else if (neighbor_dis < dismap_out[pos]) {
//                              dismap_out[pos] = neighbor_dis;
//                              anchors[j] = anchors[neighbor_i];
//                          }
//                      }
//                  }
//                  visited[j] = true;
//              }
//          }
//          // --End distance asignment.
//      }
//      
//      return;
//  }
    
    
    //          // Begin: Compute the positive distance map outside the mask.
    //          // --Begin: Find the first pixel that is not 0 (or 0). Store the coordinates in list start.
    //          long inside_start = -1;
    //          long outside_start = -1;
    //          
    //          for (long i = image_start; i < image_start + n_data; i++) {
    //              if (inside_start >= 0 && outside_start >= 0) break;
    //              if (mask_in[i] != 0) inside_start = i;
    //              if (mask_in[i] == 0) outside_start = i;
    //          }
    //          if (inside_start < 0) {
    //              cout << "Error in 'distance_map': there's no mask in the " << b + 1 << "-th image of the input array. " << endl;
    //              for (long i = image_start; i < image_start + n_data; i++) *(dismap_out + i) = 0.;
    //              continue;
    //          }
    //          // --End finding non-zero starting points.
//              float* dismap_pos = new float[n_data];
//              for (long i = 0; i < n_data; i++) dismap_pos[i] = 0;                
//              bool* visited = new bool[n_data];
//              for (long i = 0; i < n_data; i++) visited[i] = false;
//              priority_queue<Node> my_heap;
//              
//              for (long i = image_start; i < image_start + n_data; i++) {
//                  if (side == 0 && mask_in[i] != 0 || side != 0 && mask_in[i] == 0) {
//                      int* coords = new int[n_dim];
//                      long pos = i - image_start;
//                      pos2coords(pos, coords, n_dim, shape_without_batch);                
//                      struct Node inside_node = {.n_dim=n_dim, .pos=coords, .anchor=coords, .spacing=spacing_without_batch};
//                      my_heap.push(inside_node);
//                  }
//              }
////              int* start = new int[n_dim];
////              if (side == 0) pos2coords(inside_start, start, n_dim, shape_without_batch);
////              else pos2coords(outside_start, start, n_dim, shape_without_batch);
////              
////              struct Node start_node = {.n_dim=n_dim, .pos=start, .anchor=start, .spacing=spacing_without_batch};
////              my_heap.push(start_node);
//              while (!my_heap.empty()) {
//                  struct Node cur_node = my_heap.top();
//                  long cur_pos = coords2pos(cur_node.pos, n_dim, shape_without_batch);
//                  if (visited[cur_pos] && dismap_pos[cur_pos] >= cur_node.dis()) {my_heap.pop(); continue;}
//                  dismap_pos[cur_pos] = cur_node.dis();
//                  visited[cur_pos] = true;
//                  my_heap.pop();
//                  for (int d = 0; d < n_dim; d++) {
//                      for (int dir = -1; dir <= 1; dir += 2) {
//                          int* new_coords = new int[n_dim];
//                          bool out_of_bound = false;
//                          for (int k = 0; k < n_dim; k++) {
//                              if (k == d) {
//                                  int new_coord_at_dim = *(cur_node.pos + k) + dir;
//                                  if (new_coord_at_dim >= 0 && new_coord_at_dim < shape_without_batch[d])
//                                      *(new_coords + k) = new_coord_at_dim;
//                                  else out_of_bound = true;
//                              }
//                              else *(new_coords + k) = *(cur_node.pos + k);
//                          }
//                          if (out_of_bound) continue;
//                          long new_pos = coords2pos(new_coords, n_dim, shape_without_batch);
//                          int* new_anchor = new int[n_dim];
//                          float min_dis = -1.;
//                          for (int i_a = 0; i_a < 1 << (2 * n_dim); i_a++) {
//                              int* anchor_coords = new int[n_dim];
//                              bool anchor_out_of_bound = false;
//                              for (int k = 0; k < n_dim; k++) {
//                                  int dx = (i_a >> (2 * k)) % 4 - 1;
//                                  int anchor_coord_at_dim = *(cur_node.anchor + k) + dx;
//                                  if (anchor_coord_at_dim >= 0 && anchor_coord_at_dim < shape_without_batch[d])
//                                      *(anchor_coords + k) = anchor_coord_at_dim;
//                                  else anchor_out_of_bound = true;
//                              }
//                              if (anchor_out_of_bound) continue;
//                              long anchor_pos = coords2pos(anchor_coords, n_dim, shape_without_batch);
//                              if (side == 0 && mask_in[image_start + anchor_pos] == 0 || side != 0 && mask_in[image_start + anchor_pos] != 0) continue;
//                              float sum_of_sqs = 0.;
//                              for (int k = 0; k < n_dim; k++) {
//                                  float ds = (float)(anchor_coords[k] - new_coords[k]);
//                                  sum_of_sqs += ds * ds;
//                              }
//                              float cur_dis = sqrtf(sum_of_sqs);
//                              if (min_dis < 0 || cur_dis < min_dis) {
//                                  min_dis = cur_dis;
//                                  for (int k = 0; k < n_dim; k++) new_anchor[k] = anchor_coords[k];
//                              }
//                          }
//                          if (min_dis < 0) {
//                              for (int k = 0; k < n_dim; k++) new_anchor[k] = cur_node.anchor[k];
//                          }
//                          struct Node new_node = {.n_dim=n_dim, .pos=new_coords, .anchor=new_anchor, .spacing=spacing_without_batch};
//                          if (side == 0 && mask_in[image_start + new_pos] != 0 || side != 0 && mask_in[image_start + new_pos] == 0)
//                              new_node.anchor = new_coords;
//                          my_heap.push(new_node);
//                      }
//                  }
//              }
//              
//              for (long i = 0; i < n_data; i++) {
//                  dismap_out[i] += (1 - 2 * side) * dismap_pos[i];
//              }
//          }
            // --End distance asignment.
            
            // --Begin: Apply the minimal distance algorithm for the inside part.
//          float* dismap_neg = new float[n_data];
//          for (long i = 0; i < n_data; i++) dismap_neg[i] = 0;
//          pos2coords(outside_start, start, n_dim, shape_without_batch);
//  
//          for (long i = 0; i < n_data; i++) visited[i] = false;
//          start_node.pos = start;
//          start_node.anchor = start;
//          my_heap.push(start_node);
//          while (!my_heap.empty()) {
//              struct Node cur_node = my_heap.top();
//              long cur_pos = coords2pos(cur_node.pos, n_dim, shape_without_batch);
//              if (visited[cur_pos] && dismap_neg[cur_pos] >= cur_node.dis()) {my_heap.pop(); continue;}
//              dismap_neg[cur_pos] = cur_node.dis();
//              visited[cur_pos] = true;
//              my_heap.pop();
//              for (int d = 0; d < n_dim; d++) {
//                  for (int dir = -1; dir <= 1; dir += 2) {
//                      int* new_coords = new int[n_dim];
//                      bool out_of_bound = false;
//                      for (int k = 0; k < n_dim; k++) {
//                          if (k == d) {
//                              int new_coord_at_dim = *(cur_node.pos + k) + dir;
//                              if (new_coord_at_dim >= 0 && new_coord_at_dim < shape_without_batch[d])
//                                  *(new_coords + k) = new_coord_at_dim;
//                              else out_of_bound = true;
//                          }
//                          else *(new_coords + k) = *(cur_node.pos + k);
//                      }
//                      if (out_of_bound) continue;
//                      long new_pos = coords2pos(new_coords, n_dim, shape_without_batch);
//                      int* new_anchor = new int[n_dim];
//                      float min_dis = -1.;
//                      for (int i_a = 0; i_a < 1 << (2 * n_dim); i_a++) {
//                          int* anchor_coords = new int[n_dim];
//                          bool anchor_out_of_bound = false;
//                          for (int k = 0; k < n_dim; k++) {
//                              int dx = (i_a >> (2 * k)) % 4 - 1;
//                              int anchor_coord_at_dim = *(cur_node.anchor + k) + dx;
//                              if (anchor_coord_at_dim >= 0 && anchor_coord_at_dim < shape_without_batch[d])
//                                  *(anchor_coords + k) = anchor_coord_at_dim;
//                              else anchor_out_of_bound = true;
//                          }
//                          if (anchor_out_of_bound) continue;
//                          long anchor_pos = coords2pos(anchor_coords, n_dim, shape_without_batch);
//                          if (mask_in[anchor_pos] != 0) continue;
//                          float sum_of_sqs = 0.;
//                          for (int k = 0; k < n_dim; k++) {
//                              float ds = (float)(anchor_coords[k] - new_coords[k]);
//                              sum_of_sqs += ds * ds;
//                          }
//                          float cur_dis = sqrtf(sum_of_sqs);
//                          if (min_dis < 0 || cur_dis < min_dis) {
//                              min_dis = cur_dis;
//                              for (int k = 0; k < n_dim; k++) new_anchor[k] = anchor_coords[k];
//                          }
//                      }
//                      if (min_dis < 0) {
//                          for (int k = 0; k < n_dim; k++) new_anchor[k] = cur_node.anchor[k];
//                      }
//                      struct Node new_node = {.n_dim=n_dim, .pos=new_coords, .anchor=cur_node.anchor, .spacing=spacing_without_batch};
//                      if (mask_in[image_start + new_pos] == 0) new_node.anchor = new_coords;
//                      my_heap.push(new_node);
//                  }
//              }
//          }
//          // --End distance asignment.
//          
//          for (long i = 0; i < n_data; i++) {
//              dismap_out[i] = dismap_pos[i] - dismap_neg[i];
//          }

}

//string blocks = " .:;?*&@#";
//
//void show_100_100(int* map) {
//  int min = 100000, max = -100000;
//  for (int i = 0; i < 100; i++) {
//      for (int j = 0; j < 100; j++) {
//          int v = *(map + 100 * i + j);
//          if (v > max) max = v;
//          if (v < min) min = v;
//      }
//  }
//  cout << "[" << min << ", " << max << "]" << endl;
//  for (int i = 0; i < 100; i += 2) {
//      for (int j = 0; j < 100; j++) {
//          int v = *(map + 100 * i + j);
//          int ity = (int)((float)(v - min) * 8. / (float)(max - min));
//          cout << blocks[ity];
//      }
//      cout << endl;
//  }
//  int x;
//  cin >> x;
//}
//
//void show_100_100(float* map) {
//  float min = 100000., max = -100000.;
//  for (int i = 0; i < 100; i++) {
//      for (int j = 0; j < 100; j++) {
//          float v = *(map + 100 * i + j);
//          if (v > max) max = v;
//          if (v < min) min = v;
//      }
//  }
//  cout << "[" << min << ", " << max << "]" << endl;
//  for (int i = 0; i < 100; i += 2) {
//      for (int j = 0; j < 100; j++) {
//          float v = *(map + 100 * i + j);
//          int ity = (int)((v - min) * 8. / (max - min));
//          cout << blocks[ity];
//      }
//      cout << endl;
//  }
//  int x;
//  cin >> x;
//}
//
//int main() {
//  int mask[10000] = {0};
//  float dismap[10000] = {0.};
//  int shape[3] = {1, 100, 100};
//  float spacing[2] = {1.};
//  for (int i = 0; i < 100; i++) {
//      for (int j = 0; j < 100; j++) {
//          int k = 100 * i + j;
//          if ((i-50)*(i-50) + (j-50)*(j-50) <= 400)
//              mask[k] = 1;
//      }
//  }
//  //  show_100_100(mask);
//  distance_map(mask, dismap, 3, shape, spacing);
//  show_100_100(dismap);
//  return 0;
//}
