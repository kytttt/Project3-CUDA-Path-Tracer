#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#define TINYOBJLOADER_IMPLEMENTATION
#include "../third_party/tinyobjloader-release/tiny_obj_loader.h"

#define ENABLEDOF 0
using namespace std;
using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(0.f);//glm::vec3(col[0], col[1], col[2]);
			newMaterial.hasReflective = 1.f;
            newMaterial.specular.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Refractive")
        {
            const auto& col = p["RGB"];
			newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasRefractive = p["REFRACTIVE"];
            newMaterial.indexOfRefraction = p["IOR"];
			newMaterial.hasReflective = p["REFLECTIVE"];
            newMaterial.specular.color = p.contains("SPECULAR_RGB")
                ? glm::vec3(p["SPECULAR_RGB"][0], p["SPECULAR_RGB"][1], p["SPECULAR_RGB"][2])
                : glm::vec3(1.f);
            newMaterial.isThin = p.contains("THIN") ? float(p["THIN"]) : 0.f;
			newMaterial.flip = p.contains("FLIP") ? float(p["FLIP"]) : 0.f;
            newMaterial.emittance = p.contains("EMITTANCE") ? p["EMITTANCE"] : 0.f;
        }
        else if (p["TYPE"] == "Subsurface")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
			newMaterial.hasSubsurface = p["HAS_SUBSURFACE"];
            newMaterial.sigma_s = glm::vec3(p["SIGMA_S"][0], p["SIGMA_S"][1], p["SIGMA_S"][2]);
            newMaterial.sigma_a = glm::vec3(p["SIGMA_A"][0], p["SIGMA_A"][1], p["SIGMA_A"][2]);
            newMaterial.hgG = p["HG_G"];
            newMaterial.mediumScale = p.contains("MEDIUM_SCALE") ? p["MEDIUM_SCALE"] : 1.f;
			newMaterial.emittance = p.contains("EMITTANCE") ? p["EMITTANCE"] : 0.f;
        }

        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "mesh")
        {
            const auto& trans = p["TRANS"];
            const auto& rotat = p["ROTAT"];
            const auto& scale = p["SCALE"];
            glm::vec3 translation(trans[0], trans[1], trans[2]);
            glm::vec3 rotation(rotat[0], rotat[1], rotat[2]);
            glm::vec3 sc(scale[0], scale[1], scale[2]);
            glm::mat4 transform = utilityCore::buildTransformationMatrix(translation, rotation, sc);
            glm::mat4 invT = glm::inverseTranspose(transform);

            int matId = MatNameToID[p["MATERIAL"]];
            std::string file = p["FILE"];
            loadOBJ(file, matId, transform, invT);
        }
        else {


            if (type == "cube")
            {
                newGeom.type = CUBE;
            }
            else if (type == "sphere")
            {
                newGeom.type = SPHERE;
            }

            newGeom.materialid = MatNameToID[p["MATERIAL"]];
            const auto& trans = p["TRANS"];
            const auto& rotat = p["ROTAT"];
            const auto& scale = p["SCALE"];
            newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
            newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
            newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
            newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
            newGeom.inverseTransform = glm::inverse(newGeom.transform);
            newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

            geoms.push_back(newGeom);
        }
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));

    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);
    //camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    //camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
    //    2 * yscaled / (float)camera.resolution.y);

	//DOF initial setup
#if ENABLEDOF
    if (cameraData.contains("LENS_RADIUS"))
        camera.lensRadius = cameraData["LENS_RADIUS"];
    else if (cameraData.contains("APERTURE"))
        camera.lensRadius = 0.5f * (float)cameraData["APERTURE"];

    if (cameraData.contains("FOCAL_DISTANCE"))
        camera.focalDistance = cameraData["FOCAL_DISTANCE"];
    else if (cameraData.contains("FOCALPOINT"))
    {
        const auto& fp = cameraData["FOCALPOINT"];
        glm::vec3 focusPt(fp[0], fp[1], fp[2]);
        camera.focalDistance = glm::dot(focusPt - camera.position, camera.view);
    }
    if (camera.focalDistance <= 0.f)
        camera.focalDistance = glm::length(camera.lookAt - camera.position);
#endif
    /*camera.view = glm::normalize(camera.lookAt - camera.position);*/
    buildBVH();
    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

void Scene::loadOBJ(
    const std::string& path,
    int materialId,
    const glm::mat4& transform,
    const glm::mat4& invTranspose)
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> objmaterials;
    std::string warn;
    std::string err;
    std::string baseDir = path.substr(0, path.find_last_of("/\\") + 1);

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &objmaterials, &warn, &err, path.c_str(), baseDir.c_str(), true);
    if (!warn.empty()) std::cerr << "[TinyObj Warning] " << warn << std::endl;
    if (!err.empty()) std::cerr << "[TinyObj Error] " << err << std::endl;
    if (!ret)
    {
        std::cerr << "Failed to load OBJ: " << path << std::endl;
        return;
    }

    MeshGeom mg{};
    mg.materialid = materialId;
    mg.indexBegin = (int)triangles.size();
    glm::vec3 bmin(FLT_MAX), bmax(-FLT_MAX);

    for (const auto& shape : shapes)
    {
        size_t index_offset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++)
        {
            int fv = shape.mesh.num_face_vertices[f];
            if (fv != 3)
            {
                index_offset += fv;
                continue; 
            }

            Triangle tri{};
            tri.materialid = materialId;
            glm::vec3 vs[3];
            glm::vec3 ns[3];
            bool hasNormal = attrib.normals.size() > 0;
            for (int v = 0; v < 3; v++)
            {
                tinyobj::index_t idx = shape.mesh.indices[index_offset + v];
                vs[v] = glm::vec3(
                    attrib.vertices[3 * idx.vertex_index + 0],
                    attrib.vertices[3 * idx.vertex_index + 1],
                    attrib.vertices[3 * idx.vertex_index + 2]);

                if (hasNormal && idx.normal_index >= 0)
                {
                    ns[v] = glm::vec3(
                        attrib.normals[3 * idx.normal_index + 0],
                        attrib.normals[3 * idx.normal_index + 1],
                        attrib.normals[3 * idx.normal_index + 2]);
                }
            }
            index_offset += fv;

            vs[0] = glm::vec3(transform * glm::vec4(vs[0], 1.f));
            vs[1] = glm::vec3(transform * glm::vec4(vs[1], 1.f));
            vs[2] = glm::vec3(transform * glm::vec4(vs[2], 1.f));

            glm::vec3 flatN = glm::normalize(glm::cross(vs[1] - vs[0], vs[2] - vs[0]));
            if (attrib.normals.empty())
            {
                ns[0] = ns[1] = ns[2] = flatN;
            }
            else
            {

                for (int v = 0; v < 3; v++)
                {
                    ns[v] = glm::normalize(glm::vec3(invTranspose * glm::vec4(ns[v], 0.f)));
                }
            }

            tri.v0 = vs[0];
            tri.v1 = vs[1];
            tri.v2 = vs[2];
            tri.n0 = ns[0];
            tri.n1 = ns[1];
            tri.n2 = ns[2];

            for (int v = 0; v < 3; v++)
            {
                bmin = glm::min(bmin, (&tri.v0)[v]);
                bmax = glm::max(bmax, (&tri.v0)[v]);
            }
            triangles.emplace_back(tri);
        }
    }
    mg.count = (int)triangles.size() - mg.indexBegin;
    mg.bMin = bmin;
    mg.bMax = bmax;
	mg.materialid = materialId;
    meshes.emplace_back(mg);

    std::cout << "Loaded mesh: " << path
        << " tris=" << mg.count
        << " bboxMin=" << glm::to_string(bmin)
        << " bboxMax=" << glm::to_string(bmax) << std::endl;
}


struct CentroidBounds {
    glm::vec3 bmin{ FLT_MAX };
    glm::vec3 bmax{ -FLT_MAX };
    void expand(const glm::vec3& center) 
    {
        bmin = glm::min(bmin, center);
        bmax = glm::max(bmax, center);
    }
    int maxAxis() const 
    {
        glm::vec3 e = bmax - bmin;
        if (e.x > e.y && e.x > e.z) return 0;
        else if (e.y > e.x && e.y > e.z) return 1;
        else return 2;
    }
};

int buildBVHRecursive(
    const std::vector<Triangle>& tris,
    std::vector<int>& indices,
    int start, int end,
    std::vector<BVHNode>& outNodes,
    int leafThreshold)
{
    BVHNode node{};

    glm::vec3 bmin(FLT_MAX), bmax(-FLT_MAX);
    CentroidBounds cb;
    for (int i = start; i < end; ++i) 
    {
        const Triangle& t = tris[indices[i]];
        bmin = glm::min(bmin, glm::min(t.v0, glm::min(t.v1, t.v2)));
        bmax = glm::max(bmax, glm::max(t.v0, glm::max(t.v1, t.v2)));
        glm::vec3 c = (t.v0 + t.v1 + t.v2) * (1.f / 3.f);
        cb.expand(c);
    }
    node.bMin = bmin;
    node.bMax = bmax;

    int triCount = end - start;
    if (triCount <= leafThreshold) 
    {
        node.left = node.right = -1;
        node.start = start;
        node.count = triCount;
        int idx = (int)outNodes.size();
        outNodes.push_back(node);
        return idx;
    }

    int axis = cb.maxAxis();
    float midCoord = 0.5f * (cb.bmin[axis] + cb.bmax[axis]);

    int mid = start;
    for (int i = start; i < end; ++i) 
    {
        const Triangle& t = tris[indices[i]];
        glm::vec3 c = (t.v0 + t.v1 + t.v2) * (1.f / 3.f);
        if (c[axis] < midCoord)
            std::swap(indices[i], indices[mid++]);
    }
    if (mid == start || mid == end) 
    {
        mid = start + (triCount / 2);
    }

    int idx = (int)outNodes.size();
    outNodes.push_back(BVHNode{}); 

    int leftChild = buildBVHRecursive(tris, indices, start, mid, outNodes, leafThreshold);
    int rightChild = buildBVHRecursive(tris, indices, mid, end, outNodes, leafThreshold);

    node.left = leftChild;
    node.right = rightChild;
    node.start = -1;
    node.count = 0;
    outNodes[idx] = node;
    return idx;
}

void Scene::buildBVH()
{
    if (triangles.empty())
        return;

    nodeIndices.resize(triangles.size());
    std::iota(nodeIndices.begin(), nodeIndices.end(), 0);

    bvh.clear();
    bvh.reserve(triangles.size() * 2);

    const int leafThreshold = 4;
    buildBVHRecursive(triangles, nodeIndices, 0, (int)triangles.size(), bvh, leafThreshold);

    std::cout << "BVH  nodes=" << bvh.size()
        << " tris=" << triangles.size() << std::endl;
}