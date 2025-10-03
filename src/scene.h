#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadOBJ(const std::string& path,
        int materialId,
        const glm::mat4& transform,
        const glm::mat4& invTranspose);
	void buildBVH();
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<MeshGeom> meshes;
    std::vector<Triangle> triangles;
	std::vector<BVHNode> bvh;
	std::vector<int> nodeIndices;
    RenderState state;
};
