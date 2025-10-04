#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE,
    Mesh
};

struct Triangle
{
    glm::vec3 v0;
    glm::vec3 v1;
    glm::vec3 v2;
	glm::vec3 n0;
	glm::vec3 n1;
	glm::vec3 n2;
	int materialid;
};

struct MeshGeom
{
    int materialid;
    int indexBegin;
    int count;
    glm::vec3 bMin;
    glm::vec3 bMax;
};

struct BVHNode
{
    glm::vec3 bMin;
    glm::vec3 bMax;
    int left;
    int right; 
    int start;
    int count;
};

enum CubeFace {
    FACE_PX = 0,
    FACE_NX = 1,
    FACE_PY = 2,
    FACE_NY = 3,
    FACE_PZ = 4,
    FACE_NZ = 5
};

struct HostCubeMap 
{
    bool loaded = false;
    float intensity = 1.f;
    int width[6]{};
    int height[6]{};
    std::vector<glm::vec3> faces[6]; 
};

struct DeviceCubeMap 
{
    int width[6];
    int height[6];
    glm::vec3* facePtrs[6];
    float intensity;
    int hasEnv;
};

struct HostLatLongEnv 
{
    bool loaded = false;
    int width = 0;
    int height = 0;
    std::vector<glm::vec3> pixels;
    float intensity = 1.f;
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom
{
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct Material
{
    glm::vec3 color;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
    float isThin = 0.f;
    float flip =0.f;

	float hasSubsurface = 0.f;
    glm::vec3 sigma_s = glm::vec3(0.f);
    glm::vec3 sigma_a = glm::vec3(0.f);
    float hgG = 0.f;
    float mediumScale = 1.f;
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;

	//DOF parameters
    float lensRadius = 0.f;
    float focalDistance = 0.f;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
};
