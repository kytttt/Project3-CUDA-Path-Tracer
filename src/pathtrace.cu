#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
static int* dev_materialKeys = NULL;
static int* dev_indices = NULL;
static PathSegment* dev_paths_sorted = NULL;
static ShadeableIntersection* dev_intersections_sorted = NULL;
static bool sortByMaterial = true; 
static MeshGeom* dev_meshes = NULL;
static Triangle* dev_triangles = NULL;
static BVHNode* dev_bvhNodes = nullptr;
static int* dev_bvhIndices = nullptr;
static int  dev_nodeCount = 0;
#define enableAA  1
#define enableBVH 1

__constant__ DeviceCubeMap dev_envMap; // constant copy of descriptor
static glm::vec3* dev_envFaces[6] = { nullptr,nullptr,nullptr,nullptr,nullptr,nullptr };

static void uploadEnvironment(const Scene* scene) {
    if (!scene->cubemap.loaded) {
        DeviceCubeMap tmp{};
        tmp.hasEnv = 0;
        cudaMemcpyToSymbol(dev_envMap, &tmp, sizeof(DeviceCubeMap));
        return;
    }
    DeviceCubeMap desc{};
    desc.intensity = scene->cubemap.intensity;
    desc.hasEnv = 1;
    for (int i = 0; i < 6; ++i) {
        int count = (int)scene->cubemap.faces[i].size();
        if (count == 0) { desc.hasEnv = 0; continue; }
        cudaMalloc(&dev_envFaces[i], count * sizeof(glm::vec3));
        cudaMemcpy(dev_envFaces[i], scene->cubemap.faces[i].data(),
            count * sizeof(glm::vec3), cudaMemcpyHostToDevice);
        desc.facePtrs[i] = dev_envFaces[i];
        desc.width[i] = scene->cubemap.width[i];
        desc.height[i] = scene->cubemap.height[i];
    }
    cudaMemcpyToSymbol(dev_envMap, &desc, sizeof(DeviceCubeMap));
}



void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
    cudaMalloc(&dev_materialKeys, pixelcount * sizeof(int));
    cudaMalloc(&dev_indices, pixelcount * sizeof(int));
    cudaMalloc(&dev_paths_sorted, pixelcount * sizeof(PathSegment));
    cudaMalloc(&dev_intersections_sorted, pixelcount * sizeof(ShadeableIntersection));

    if (!scene->meshes.empty())
    {
        cudaMalloc(&dev_meshes, scene->meshes.size() * sizeof(MeshGeom));
        cudaMemcpy(dev_meshes, scene->meshes.data(), scene->meshes.size() * sizeof(MeshGeom), cudaMemcpyHostToDevice);
    }
    if (!scene->triangles.empty())
    {
        cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
        cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
    }
    if (!scene->bvh.empty()) 
    {
        dev_nodeCount = (int)scene->bvh.size();
        cudaMalloc(&dev_bvhNodes, dev_nodeCount * sizeof(BVHNode));
        cudaMemcpy(dev_bvhNodes, scene->bvh.data(),
            dev_nodeCount * sizeof(BVHNode), cudaMemcpyHostToDevice);

        cudaMalloc(&dev_bvhIndices, scene->nodeIndices.size() * sizeof(int));
        cudaMemcpy(dev_bvhIndices, scene->nodeIndices.data(),
            scene->nodeIndices.size() * sizeof(int), cudaMemcpyHostToDevice);
    }
	uploadEnvironment(scene);
    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
    cudaFree(dev_materialKeys);
    cudaFree(dev_indices);
    cudaFree(dev_paths_sorted);
    cudaFree(dev_intersections_sorted);
    cudaFree(dev_meshes);
    cudaFree(dev_triangles);
    cudaFree(dev_bvhNodes);
    cudaFree(dev_bvhIndices);
    for (int i = 0; i < 6; ++i) {
        if (dev_envFaces[i]) {
            cudaFree(dev_envFaces[i]);
            dev_envFaces[i] = nullptr;
        }
    }
    checkCUDAError("pathtraceFree");
}

__host__ __device__ inline glm::vec2 concentricDisk(float u1, float u2)
{
    float sx = 2.f * u1 - 1.f;
    float sy = 2.f * u2 - 1.f;

    if (sx == 0.f && sy == 0.f) return glm::vec2(0);

    float r, theta;
    if (fabsf(sx) > fabsf(sy)) {
        r = sx;
        theta = (PI / 4.f) * (sy / sx);
    }
    else {
        r = sy;
        theta = (PI / 2.f) - (PI / 4.f) * (sx / sy);
    }
    return r * glm::vec2(cosf(theta), sinf(theta));
}

__device__ void traverseBVHTriangles(
    const Ray& ray,
    const BVHNode* nodes,
    const int* triIndices,
    const Triangle* tris,
    float& t_min,
    int& hitMaterial,
    glm::vec3& hitNormal)
{
    if (!nodes) return;
    int stack[64];
    int sp = 0;
    stack[sp++] = 0; 

    while (sp) {
        int ni = stack[--sp];
        const BVHNode& node = nodes[ni];

        float hb = aabbIntersectionTest(ray, node.bMin, node.bMax);
        if (hb < 0.f || hb > t_min) continue;

        if (node.count > 0) {
            for (int i = 0; i < node.count; ++i) {
                int ti = triIndices[node.start + i];
                const Triangle& tri = tris[ti];
                glm::vec3 tmpP, tmpN;
                float t = triangleIntersectionTest(tri, ray, tmpP, tmpN);
                if (t > 0.f && t < t_min) {
                    t_min = t;
                    hitMaterial = tri.materialid;
                    hitNormal = tmpN;
                }
            }
        }
        else {
 
            if (node.left >= 0)  stack[sp++] = node.left;
            if (node.right >= 0) stack[sp++] = node.right;
        }
    }
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // TODO: implement antialiasing by jittering the ray
        float jx = 0.f;
        float jy = 0.f;
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0.f, 1.f);
#if enableAA
            jx = u01(rng); 
            jy = u01(rng);
#endif
        float px = ((float)x + jx) - (float)cam.resolution.x * 0.5f;
        float py = ((float)y + jy) - (float)cam.resolution.y * 0.5f;

        //segment.ray.direction = glm::normalize(
        //    cam.view
        //    - cam.right * cam.pixelLength.x * px
        //    - cam.up * cam.pixelLength.y * py
        //);
        glm::vec3 originalDir = glm::normalize(
            cam.view
            - cam.right * cam.pixelLength.x * px
            - cam.up * cam.pixelLength.y * py
        );

        glm::vec3 origin = cam.position;
        glm::vec3 direction = originalDir;

        if (cam.lensRadius > 0.f)
        {
            float projectDis = glm::dot(originalDir, glm::normalize( cam.view));
			if (projectDis == 0.f) projectDis = 1.f;
            float ft = cam.focalDistance / projectDis;
            glm::vec3 focalPoint = cam.position + originalDir * ft;

            glm::vec2 randOffset = concentricDisk(u01(rng), u01(rng)) * cam.lensRadius;

            origin = cam.position + randOffset.x * cam.right + randOffset.y * cam.up;
            direction = glm::normalize(focalPoint - origin);
        }

        segment.ray.origin = origin;
        segment.ray.direction = direction;
        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections,
    MeshGeom* meshes,
    int meshCount,
    Triangle* triangles,
    int triangleCount)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool hitMesh = false;
		int hit_material = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
				hit_material = geom.materialid;
            }
        }

        for (int m = 0; m < meshCount; ++m)
        {
            MeshGeom& mg = meshes[m];
            float boxT = 0.f;
            boxT = aabbIntersectionTest(pathSegment.ray, mg.bMin, mg.bMax);
            if (boxT < 0.f)
                continue;
            for (int tIdx = 0; tIdx < mg.count; ++tIdx)
            {
                const Triangle& tri = triangles[mg.indexBegin + tIdx];
                float t = triangleIntersectionTest(tri, pathSegment.ray, tmp_intersect, tmp_normal);
                if (t > 0.f && t < t_min)
                {
                    t_min = t;
					hitMesh = true;
                    hit_material = tri.materialid;
                    normal = tmp_normal;
                }
            }
        }

        if (hit_material == -1 )
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = hit_material;
            intersections[path_index].surfaceNormal = normal;
        }
    }
}

__global__ void computeIntersectionsBVH(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections,
    const BVHNode* bvh,
    int nodeCount,
    const int* bvhIndices,
    const Triangle* triangles)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_index >= num_paths) return;

    PathSegment pathSegment = pathSegments[path_index];

    float t_min = FLT_MAX;
    int hit_material = -1;
    glm::vec3 normal(0.f);
    bool outside = true;
    glm::vec3 tmp_intersect, tmp_normal;

    for (int i = 0; i < geoms_size; i++)
    {
        Geom& g = geoms[i];
        float t = -1.f;
        if (g.type == CUBE) {
            t = boxIntersectionTest(g, pathSegment.ray, tmp_intersect, tmp_normal, outside);
        }
        else if (g.type == SPHERE) {
            t = sphereIntersectionTest(g, pathSegment.ray, tmp_intersect, tmp_normal, outside);
        }
        if (t > 0.f && t < t_min) {
            t_min = t;
            hit_material = g.materialid;
            normal = tmp_normal;
        }
    }
    
    traverseBVHTriangles(pathSegment.ray, bvh, bvhIndices, triangles, t_min, hit_material, normal);

    if (hit_material == -1) {
        intersections[path_index].t = -1.f;
    }
    else {
        intersections[path_index].t = t_min;
        intersections[path_index].materialId = hit_material;
        intersections[path_index].surfaceNormal = normal;
    }
}

__device__ inline glm::vec3 sampleCubemapDir(const DeviceCubeMap& env, const glm::vec3& d) {
    if (!env.hasEnv) return BACKGROUND_COLOR;
    glm::vec3 dir = glm::normalize(d);
    float ax = fabsf(dir.x), ay = fabsf(dir.y), az = fabsf(dir.z);
    int face;
    float u, v;
    if (ax >= ay && ax >= az) { // X major
        if (dir.x > 0) { face = FACE_PX; u = -dir.z / ax; v = -dir.y / ax; }
        else { face = FACE_NX; u = dir.z / ax; v = -dir.y / ax; }
    }
    else if (ay >= ax && ay >= az) { // Y major
        if (dir.y > 0) { face = FACE_PY; u = dir.x / ay; v = dir.z / ay; }
        else { face = FACE_NY; u = dir.x / ay; v = -dir.z / ay; }
    }
    else { // Z major
        if (dir.z > 0) { face = FACE_PZ; u = dir.x / az; v = -dir.y / az; }
        else { face = FACE_NZ; u = -dir.x / az; v = -dir.y / az; }
    }
    // map from [-1,1] to [0,1]
    u = 0.5f * (u + 1.f);
    v = 0.5f * (v + 1.f);

    int w = env.width[face];
    int h = env.height[face];
    if (w <= 0 || h <= 0) return BACKGROUND_COLOR;

    float fx = u * (w - 1);
    float fy = v * (h - 1);
    int x0 = (int)floorf(fx);
    int y0 = (int)floorf(fy);
    int x1 = glm::min(x0 + 1, w - 1);
    int y1 = glm::min(y0 + 1, h - 1);
    float tx = fx - x0;
    float ty = fy - y0;

    const glm::vec3* facePtr = env.facePtrs[face];
    glm::vec3 c00 = facePtr[y0 * w + x0];
    glm::vec3 c10 = facePtr[y0 * w + x1];
    glm::vec3 c01 = facePtr[y1 * w + x0];
    glm::vec3 c11 = facePtr[y1 * w + x1];
    glm::vec3 cx0 = c00 * (1 - tx) + c10 * tx;
    glm::vec3 cx1 = c01 * (1 - tx) + c11 * tx;
    return (cx0 * (1 - ty) + cx1 * ty) * env.intensity;
}


// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                pathSegments[idx].color *= u01(rng); // apply some noise because why not
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
        }
    }
}

__global__ void shadeDiffuseBSDF(
    int iter,
    int depth,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    PathSegment& seg = pathSegments[idx];
    if (seg.remainingBounces <= 0) return;

    ShadeableIntersection isect = shadeableIntersections[idx];
    if (isect.t < 0.f)
    {
        //seg.color *= BACKGROUND_COLOR;
        //seg.remainingBounces = 0;
        DeviceCubeMap cube;
        memcpy(&cube, &dev_envMap, sizeof(DeviceCubeMap));

        glm::vec3 L(0.f);
        if (cube.hasEnv) {
            L = sampleCubemapDir(cube, seg.ray.direction);
        }
        else
			L = BACKGROUND_COLOR;
        seg.color *= L;
        seg.remainingBounces = 0;
        return;
        return;
    }

    const Material& m = materials[isect.materialId];

    if (m.emittance > 0.f)
    {
        seg.color *= (m.color * m.emittance);
        seg.remainingBounces = 0;
        return;
    }

    thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
    glm::vec3 hitPoint = getPointOnRay(seg.ray, isect.t);
    glm::vec3 n = glm::normalize(isect.surfaceNormal);

    scatterRay(seg, hitPoint, n, m, rng);
}

__global__ void buildMaterialKeys(
    int n,
    const ShadeableIntersection* intersections,
    const PathSegment* paths,
    int* keys,
    int* indices)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const ShadeableIntersection& s = intersections[i];

    if ((paths[i].remainingBounces <= 0) || (s.t < 0.f)) {
		keys[i] = INT_MAX;
    }
    else
		keys[i] = s.materialId;
    indices[i] = i;
}


__global__ void sortByIndices(
    int n,
    const int* indices,
    const PathSegment* ipaths,
    const ShadeableIntersection* iIntersections,
    PathSegment* opaths,
    ShadeableIntersection* oIntersections)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    opaths[i] = ipaths[indices[i]];
    oIntersections[i] = iIntersections[indices[i]];
}

struct BounceEnd
{
    __host__ __device__
        bool operator()(const PathSegment& p) const
    {
        return p.remainingBounces <= 0;
    }
};

__global__ void handleEnd(int n, glm::vec3* image, PathSegment* paths)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    PathSegment& p = paths[i];
    if (p.remainingBounces <= 0 && p.pixelIndex >= 0)
    {
        image[p.pixelIndex] += p.color;

        p.pixelIndex = -1;
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
	bool bvhEnabled = enableBVH;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        if (!bvhEnabled) {
            computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                depth,
                num_paths,
                dev_paths,
                dev_geoms,
                hst_scene->geoms.size(),
                dev_intersections,
                dev_meshes,
                hst_scene->meshes.size(),
                dev_triangles,
                hst_scene->triangles.size()
                );
        }
        else {
            computeIntersectionsBVH <<<numblocksPathSegmentTracing, blockSize1d >> > (
                depth,
                num_paths,
                dev_paths,
                dev_geoms,
                (int)hst_scene->geoms.size(),
                dev_intersections,
                dev_bvhNodes,
                dev_nodeCount,
                dev_bvhIndices,
                dev_triangles
                );
        }
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        //shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
        //    iter,
        //    num_paths,
        //    dev_intersections,
        //    dev_paths,
        //    dev_materials
        //);
                // Shade using diffuse BSDF

                // After intersections are computed
        if (sortByMaterial)
        {
            buildMaterialKeys <<<numblocksPathSegmentTracing, blockSize1d >>> (
                num_paths,
                dev_intersections,
                dev_paths,
                dev_materialKeys,
                dev_indices);
            
            thrust::sort_by_key(thrust::device, dev_materialKeys, dev_materialKeys + num_paths, dev_indices);
            
            sortByIndices <<<numblocksPathSegmentTracing, blockSize1d >>> (
                num_paths,
                dev_indices,
                dev_paths,
                dev_intersections,
                dev_paths_sorted,
                dev_intersections_sorted);
            
            std::swap(dev_paths, dev_paths_sorted);
            std::swap(dev_intersections, dev_intersections_sorted);
        }

        shadeDiffuseBSDF <<<numblocksPathSegmentTracing, blockSize1d >>> (
            iter,
            depth,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials
            );
        checkCUDAError("shade diffuse");
        cudaDeviceSynchronize();
        //iterationComplete = true; // TODO: should be based off stream compaction results.
        handleEnd <<<numblocksPathSegmentTracing, blockSize1d >>> (
            num_paths, dev_image, dev_paths);

        PathSegment* dev_new_end = thrust::remove_if(thrust::device, dev_paths, dev_paths + num_paths, BounceEnd());
        num_paths = dev_new_end - dev_paths;
		if (num_paths <= 0) iterationComplete = true;
        depth++;
        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }

        if(depth >= traceDepth) {
            iterationComplete = true;
		}
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
