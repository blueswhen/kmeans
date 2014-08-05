// Copyright 2014-4 sxniu
#include "include/utils.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include <opencv/highgui.h>
#include <vector>
#include <stack>
#include <queue>

#include "include/colour.h"
#include "include/ImageData.h"

#define COMPONENTS 3
#define WASHED 0xffffffff
#define IN_QUEUE -2
#define EPSILON 0.0001

typedef unsigned char uchar;

namespace utils {

void ReadImage(const char* file_name, ImageData<int>* image_data) {
  if (!image_data->IsEmpty()) {
    printf("error: image data must be empty");
    return;
  }
  image_data->m_file_name = file_name;
  int& width = image_data->m_width;
  int& height = image_data->m_height;
  std::vector<int>* data = image_data->m_data;

  IplImage* cv_image = cvLoadImage(file_name);

  width = cv_image->width;
  height = cv_image->height;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int index = y * cv_image->widthStep + x * COMPONENTS;
      uchar* cv_data = reinterpret_cast<uchar*>(cv_image->imageData);
      int colour = (static_cast<int>(cv_data[index + 2]) << 16) +
                   (static_cast<int>(cv_data[index + 1]) << 8) +
                   (static_cast<int>(cv_data[index]));
      data->push_back(colour);
    }
  }
}

void TurnGray(const ImageData<int>& input_image, ImageData<uchar>* gray_image) {
  if (input_image.IsEmpty()) {
    printf("error: input image data is empty");
    return;
  }
  int height = input_image.GetHeight();
  int width = input_image.GetWidth();

  if (gray_image->IsEmpty()) {
    gray_image->CreateEmptyImage(width, height);
  }
  for (int y = 0; y < height; ++y) {
    for (int x  = 0; x < width; ++x) {
      int index = y * width + x;
      int red = (GET_PIXEL(&input_image, index) & RED) >> 16;
      int green = (GET_PIXEL(&input_image, index) & GREEN) >> 8;
      int blue = GET_PIXEL(&input_image, index) & BLUE;
      uchar gray = static_cast<uchar>(red * 0.3 + green * 0.59 + blue * 0.11);
      SET_PIXEL(gray_image, index, gray);
    }
  }
}

void SaveImage(const char* out_file_name, const ImageData<int>& image_data) {
  if (image_data.IsEmpty()) {
    printf("error: image data is empty");
    return;
  }
  int width = image_data.GetWidth();
  int height = image_data.GetHeight();

  CvSize size;
  size.width = width;
  size.height = height;
  IplImage* cv_image = cvCreateImage(size, 8, COMPONENTS);
  if (cv_image == NULL) {
    printf("error: the creation of cv image is failure");
    return;
  }

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int index = y * cv_image->widthStep + x * COMPONENTS;
      uchar* cv_data = reinterpret_cast<uchar*>(cv_image->imageData);
      int colour = GET_PIXEL(&image_data, y * width + x);
      cv_data[index + 2] = static_cast<uchar>((colour & RED) >> 16);
      cv_data[index + 1] = static_cast<uchar>((colour & GREEN) >> 8);
      cv_data[index] = static_cast<uchar>(colour & BLUE);
    }
  }
  cvSaveImage(out_file_name, cv_image);
}

void SaveImage(const char* out_file_name, const ImageData<uchar>& image_data) {
  if (image_data.IsEmpty()) {
    printf("error: image data is empty");
    return;
  }
  int width = image_data.GetWidth();
  int height = image_data.GetHeight();

  CvSize size;
  size.width = width;
  size.height = height;
  IplImage* cv_image = cvCreateImage(size, 8, COMPONENTS);
  if (cv_image == NULL) {
    printf("error: the creation of cv image is failure");
    return;
  }

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int index = y * cv_image->widthStep + x * COMPONENTS;
      uchar* cv_data = reinterpret_cast<uchar*>(cv_image->imageData);
      uchar gray = GET_PIXEL(&image_data, y * width + x);
      cv_data[index + 2] = gray;
      cv_data[index + 1] = gray;
      cv_data[index] = gray;
    }
  }
  cvSaveImage(out_file_name, cv_image);
}

void GetGradiendMap(const ImageData<uchar>& gray_image, ImageData<uchar>* grad_image) {
  int image_width = gray_image.GetWidth();
  int image_height = gray_image.GetHeight();

  if (grad_image != NULL && grad_image->IsEmpty()) {
    grad_image->CreateEmptyImage(image_width, image_height);
  } else {
    printf("error: grad_image should be an empty image\n");
  }

  for (int y = 0; y < image_height; ++y) {
    for (int x = 0; x < image_width; ++x) {
      int index_cen = y * image_width + x;
      if (x == 0 || x == image_width - 1 ||
          y == 0 || y == image_height - 1) {
        SET_PIXEL(grad_image, index_cen, 255);
        continue;
      }
      int index[8] = EIGHT_ARROUND_POSITION(x, y, image_width, image_height);

      double gx = GET_PIXEL(&gray_image, index[3]) +
                  2 * GET_PIXEL(&gray_image, index[2]) +
                  GET_PIXEL(&gray_image, index[1]) +
                  -GET_PIXEL(&gray_image, index[5]) +
                  - 2 * GET_PIXEL(&gray_image, index[6]) +
                  - GET_PIXEL(&gray_image, index[7]);

      double gy = GET_PIXEL(&gray_image, index[7]) +
                  2 * GET_PIXEL(&gray_image, index[0]) +
                  GET_PIXEL(&gray_image, index[1]) +
                  - GET_PIXEL(&gray_image, index[5]) +
                  - 2 * GET_PIXEL(&gray_image, index[4]) +
                  - GET_PIXEL(&gray_image, index[3]);

      double sum_of_squares = pow(gx, 2) + pow(gy, 2);
	    uchar dst_gray = std::min(static_cast<int>(sqrt(sum_of_squares)), 255);
      // dst_gray = TURN_COORDINATE_TO_COLOUR(dst_gray, dst_gray, dst_gray);
      SET_PIXEL(grad_image, index_cen, dst_gray);
    }
  }
}

void DoMarkConnectedArea(const ImageData<uchar>& grad_image, ImageData<int>* marked_image,
                         int x, int y, int width, int height, int mark_num,
                         int max_threshold) {

  std::stack<int> unsearched_points;
  unsearched_points.push(y * width + x);
  while (unsearched_points.size() != 0) {
    int index = unsearched_points.top(); 
    unsearched_points.pop();
    int y = index / width;
    int x = index - y * width;
    int arrounds[8] = EIGHT_ARROUND_POSITION(x, y, width, height);
    for (int i = 0; i < 8; ++i) {
      int gradient = static_cast<int>(GET_PIXEL(&grad_image, arrounds[i]));
      int mark_value = GET_PIXEL(marked_image, arrounds[i]);

      assert(mark_value == 0 || std::abs(mark_value) == mark_num);
      if (mark_value == 0) {
        if (gradient <= max_threshold) {
          SET_PIXEL(marked_image, arrounds[i], mark_num);
          unsearched_points.push(arrounds[i]);
        } else {
          SET_PIXEL(marked_image, index, -mark_num);
        }
      }
    }
  }
}

void MarkConnectedArea(const ImageData<uchar>& grad_image, ImageData<int>* marked_image,
                       int max_threshold) {
  if (grad_image.IsEmpty()) {
    printf("error: the grad_image is empty\n");
    return;
  }
  int width = grad_image.GetWidth();
  int height = grad_image.GetHeight();
  if (marked_image->IsEmpty()) {
    marked_image->CreateEmptyImage(width, height);
  }

  int mark_num = 100;
  for (int y = 1; y < height - 1; ++y) {
    for (int x = 1; x < width - 1; ++x) {
      int index = y * width + x;
      int gradient = static_cast<int>((*grad_image.m_data)[index]);
      int mark_value = (*marked_image->m_data)[index];
      if (mark_value == 0 && gradient <= max_threshold) {
        SET_PIXEL(marked_image, index, mark_num);
        DoMarkConnectedArea(grad_image, marked_image, x, y, width, height, mark_num,
                            max_threshold);
        // mark_num++;
        mark_num += 10000;
      }
    }
  }
}

void Watershed(const ImageData<uchar>& grad_image, ImageData<int>* marked_image, int start_gradient) {
  if (grad_image.IsEmpty()) {
    printf("error: the grad_image is empty\n");
    return;
  }
  int width = grad_image.GetWidth();
  int height = grad_image.GetHeight();
  std::queue<int> grad_queues[256];
  for (int y = 1; y < height - 1; ++y) {
    for (int x = 1; x < width - 1; ++x) {
      int index = y * width + x;
      int mark_value = (*marked_image->m_data)[index];
      if (mark_value < 0 && mark_value != IN_QUEUE) {
        SET_PIXEL(marked_image, index, -mark_value);
        int arrounds[8] = EIGHT_ARROUND_POSITION(x, y, width, height);
        for (int i = 0; i < 8; i += 2) {
          int arround_mark_value = (*marked_image->m_data)[arrounds[i]];
          if (arround_mark_value == 0) {
            int grad_value = (*grad_image.m_data)[arrounds[i]];
            assert(grad_value > start_gradient && grad_value < 256);
            grad_queues[grad_value].push(arrounds[i]);
            SET_PIXEL(marked_image, arrounds[i], IN_QUEUE);
          }
        }
      }
    }
  }

  int start_idx = start_gradient;
  for (; start_idx < 256; ++start_idx) {
    if (!grad_queues[start_idx].empty()) {
      break;
    }
  }
  int queues_idx = start_idx;
  while(true) {
    if (grad_queues[queues_idx].empty()) {
      if(++queues_idx >= 256) {
        break;
      }
      continue;
    }
    int mark_index = grad_queues[queues_idx].front();
    grad_queues[queues_idx].pop();
    int mark_value = GET_PIXEL(marked_image, mark_index);
    if (mark_value != IN_QUEUE) {
      continue;
    }

    int mark_number = 0;
    int mark_y = mark_index / width;
    int mark_x = mark_index - mark_y * width;
    int mark_arrounds[8] = EIGHT_ARROUND_POSITION(mark_x, mark_y, width, height);
    for (int i = 0; i < 8; i += 2) {
      int mark_value = GET_PIXEL(marked_image, mark_arrounds[i]);
      if (mark_value == WASHED || mark_value == IN_QUEUE) {
        continue;
      }

      if (mark_value == 0) {
        int grad_value = static_cast<int>((*grad_image.m_data)[mark_arrounds[i]]);
        assert(grad_value > start_gradient);
        grad_queues[grad_value].push(mark_arrounds[i]);
        queues_idx = std::min(queues_idx, grad_value);
        SET_PIXEL(marked_image, mark_arrounds[i], IN_QUEUE);
      } else {
        if (mark_number == 0) {
          mark_number = mark_value;
          SET_PIXEL(marked_image, mark_index, mark_number);
        } else if (mark_number != mark_value) {
          SET_PIXEL(marked_image, mark_arrounds[i], WASHED);
        }
      }
    }
  }
}

// [start, end)
int GenRanNumI(int seed, int start, int end) {
  srand(seed);
  return (rand() % (end - start)) + start;
}

double GenRanNumD(int seed, int start, int end) {
  srand(seed);
  int integer = (rand() % (end - start)) + start;
  double decimals = rand() / static_cast<double>(RAND_MAX);
  return integer + decimals;
}

void generateCentersPP(const ImageData<uchar>& gray_image,
                       double* centers, int n,
                       int k, int trials) {
  int N = n;
  int K = k;
  int seed = gray_image.GetRandomSeed();
  centers[0] = GenRanNumI(seed, 0, N);

  int sum = 0;
  std::vector<int> dist(3 * N);
  int* dist1 = &dist[0];
  int* dist2 = &dist[N];
  int* dist3 = &dist[2 * N];

  const int& center = static_cast<int>(GET_PIXEL(&gray_image, centers[0]));
  for(int i = 0; i < N; i++) {
    const int& data = static_cast<int>(GET_PIXEL(&gray_image, i));
    dist[i] = (data - center) * (data - center);
    sum += dist[i];
  }

  for(int cluster = 1; cluster < K; ++cluster) {
    int bestSum = RAND_MAX;
    int bestCenter = -1;

    for(int j = 0; j < trials; j++) {
      double p = GenRanNumD(++seed, 0, 1) * sum;
      int ci = 0;
      for(int i = 0; i < N - 1; i++) {
        if((p -= dist1[i]) <= 0) {
          ci = i;
          break;
        }
      }

      const int& center_ci = static_cast<int>(GET_PIXEL(&gray_image, ci));
      int s = 0;
      for(int i = 0; i < N; i++) {
        const int& data = static_cast<int>(GET_PIXEL(&gray_image, i));
        dist2[i] = std::min((data - center_ci) * (data - center_ci), dist1[i]);
        s += dist2[i];
      }

      if(s < bestSum) {
        bestSum = s;
        bestCenter = ci;
        std::swap(dist2, dist3);
      }
    }
    centers[cluster] = bestCenter;
    sum = bestSum;
    std::swap(dist1, dist3);
  }

  for(int cluster = 0; cluster < K; ++cluster) {
    centers[cluster] = GET_PIXEL(&gray_image, centers[cluster]);
  }
}

void KMeansDistanceComputer(const ImageData<uchar>& gray_image, ImageData<int>* marked_image,
                            const double* centers, int k, int n) {
  int K = k;
  int N = n;
  for(int i = 0; i < N; ++i) {
    int sample = static_cast<int>(GET_PIXEL(&gray_image, i));
    int cluster_best = 0;
    double min_dist = DBL_MAX;

    for(int cluster = 0; cluster < K; ++cluster) {
      const double& center = centers[cluster];
      double dist = (sample - center) * (sample - center);

      if(dist < min_dist) {
        min_dist = dist;
        cluster_best = cluster;
      }
    }
    SET_PIXEL(marked_image, i, cluster_best);
  }
}

void Kmeans(const ImageData<uchar>& gray_image, ImageData<int>* marked_image, int k, int iter) {
  int width = gray_image.GetWidth();
  int height = gray_image.GetHeight();
  int N = width * height;
  int K = k;
  int times = K != 1 ? iter : 2;
  if (marked_image->IsEmpty()) {
    marked_image->CreateEmptyImage(width, height);
  }

  std::vector<double> centers_container(2 * K, 0);
  double* centers = &centers_container[0];
  double* old_centers = &centers_container[K];
  for (int j = 0; j < times; ++j) {
    std::vector<int> counters(K, 0);
    double max_center_shift = DBL_MAX;
    if (j == 0) {
      // Arthur & Vassilvitskii (2007) k-means++: The Advantages of Careful Seeding
      int trials = 3;
      generateCentersPP(gray_image, centers, N, K, trials);
    } else {
      max_center_shift = 0;
      for(int i = 0; i < N; ++i) {
        int sample = static_cast<int>(GET_PIXEL(&gray_image, i));
        int k_value = GET_PIXEL(marked_image, i);
        assert(k_value >= 0 && k_value < K);
        centers[k_value] += sample;
        counters[k_value]++;
      }

      for(int cluster = 0; cluster < K; cluster++) {
        if(counters[cluster] != 0)
            continue;

        // if some cluster appeared to be empty then:
        //   1. find the biggest cluster
        //   2. find the farthest from the center point in the biggest cluster
        //   3. exclude the farthest point from the biggest cluster and form a new 1-point cluster.
        int max_cluster = 0;
        for(int i = 1; i < K; ++i) {
          if(counters[max_cluster] < counters[i]) {
            max_cluster = i;
          }
        }

        double max_dist = 0;
        int farthest_i = -1;
        double mean_max_cluster = centers[max_cluster] / counters[max_cluster];

        for(int i = 0; i < N; i++) {
          int k = GET_PIXEL(marked_image, i);
          if(k != max_cluster)
            continue;
          int sample = static_cast<int>(GET_PIXEL(&gray_image, i));
          double dist = (sample - mean_max_cluster) * (sample - mean_max_cluster);

          if(max_dist <= dist) {
            max_dist = dist;
            farthest_i = i;
          }
        }

        counters[max_cluster]--;
        counters[cluster]++;
        SET_PIXEL(marked_image, farthest_i, cluster);

        int data = static_cast<int>(GET_PIXEL(&gray_image, farthest_i));
        centers[max_cluster] -= data;
        centers[cluster] += data;
      }

      for(int cluster = 0; cluster < K; ++cluster) {
         centers[cluster] /= counters[cluster];
         if(j > 0) {
           double t = centers[cluster] - old_centers[cluster];
           double dist = t * t;
           max_center_shift = std::max(max_center_shift, dist);
         }
      }
    }

    if (max_center_shift <= EPSILON) {
      break;
    }

    KMeansDistanceComputer(gray_image, marked_image, centers, K, N);
    std::swap(centers, old_centers);
    for (int cluster = 0; cluster < K; ++cluster) {
      centers[cluster] = 0;
    }
  }
}

void ShowMarkedImage(ImageData<int>* marked_image) {
  int width = marked_image->GetWidth();
  int height = marked_image->GetHeight();
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int index = y * width + x;
      int data = GET_PIXEL(marked_image, index);
      SET_PIXEL(marked_image, index, data * 10000 + 100);
    }
  }
}

}  // namespace utils
