#pragma once

#include "sceneStructs.h"
#include <vector>
#include <json.hpp>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadOBJ(const std::string& path,
        int materialId,
        const glm::mat4& transform,
        const glm::mat4& invTranspose);
	void buildBVH();
    bool loadLatLongEnvironment(const nlohmann::json& envJson);
    bool convertLatLongToCubemap(int faceRes);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<MeshGeom> meshes;
    std::vector<Triangle> triangles;
	std::vector<BVHNode> bvh;
	std::vector<int> nodeIndices;
    HostCubeMap cubemap;
    RenderState state;
    HostLatLongEnv latlongEnv;
};
