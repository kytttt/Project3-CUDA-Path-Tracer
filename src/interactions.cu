#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>
#include <glm/gtx/norm.hpp>

#define enableRussianRoulette 1;

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}


__host__ __device__ inline float reflectance(float cosine, float etaI, float etaT)
{
    
    cosine = fminf(fmaxf(cosine, 0.f), 1.f);
    float r0 = (etaI - etaT) / (etaI + etaT);
    r0 = r0 * r0;
    return r0 + (1.f - r0) * powf((1.f - cosine), 5.f);
}

__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    thrust::uniform_real_distribution<float> u01(0.f, 1.f);
    const float EPS = 1e-4f;

    glm::vec3 inVec = glm::normalize(pathSegment.ray.direction);
    glm::vec3 n = glm::normalize(normal);

    if (m.hasRefractive > 0.f && m.indexOfRefraction > 0.f)
    {
        bool in_or_out = glm::dot(inVec, n) < 0.f;
        glm::vec3 nDir = in_or_out ? n : -n;

        float etaI = in_or_out ? 1.f : m.indexOfRefraction;
        float etaT = in_or_out ? m.indexOfRefraction : 1.f;
        float eta_ratio = etaI / etaT; 

        float consine = fminf(fmaxf(-glm::dot(inVec, nDir), 0.f), 1.f);
        float sin2 = eta_ratio * eta_ratio * fmaxf(0.f, 1.f - consine * consine);
		bool cannotRefract = sin2 > 1.f;

        float F = reflectance(consine, etaI, etaT);

        glm::vec3 newDir;
        float rand_f = u01(rng);

        if (cannotRefract || rand_f < F)
        {
            newDir = glm::reflect(inVec, nDir);
            pathSegment.color *= m.specular.color;
            pathSegment.ray.origin = intersect + nDir * EPS;
        }
        else
        {

                newDir = glm::refract(inVec, nDir, eta_ratio);

                if (glm::length2(newDir) < 1e-10f)
                {
                    newDir = glm::reflect(inVec, nDir);
                    pathSegment.color *= m.specular.color;
                    pathSegment.ray.origin = intersect + nDir * EPS;
                }
                else
                {
                    glm::vec3 transTint = m.color;

                    pathSegment.color *= transTint;
                    pathSegment.ray.origin = intersect - nDir * EPS;
                }

        }

        pathSegment.ray.direction = glm::normalize(newDir);
        pathSegment.remainingBounces--;
        return;
    }

    float diffuseStrength = (m.color.x + m.color.y + m.color.z) * (1.f / 3.f);
    float specStrength = 0.f;
    if (m.hasReflective > 0.f)
        specStrength = (m.specular.color.x + m.specular.color.y + m.specular.color.z) * (1.f / 3.f);
    float sum = diffuseStrength + specStrength;
    float pSpec = (sum > 0.f && specStrength > 0.f) ? specStrength / sum : 0.f;

    glm::vec3 wi;
    float xi2 = u01(rng);
    if (xi2 < pSpec)
    {
        wi = glm::reflect(inVec, n);
        pathSegment.color *= (m.specular.color / (pSpec > 0.f ? pSpec : 1.f));
        pathSegment.ray.origin = intersect + n * EPS;
    }
    else
    {
        wi = calculateRandomDirectionInHemisphere(n, rng);
        if (glm::dot(wi, n) < 0.f) wi = -wi;
        float pDiffuse = 1.f - pSpec;
        pathSegment.color *= (m.color / (pDiffuse > 0.f ? pDiffuse : 1.f));
        pathSegment.ray.origin = intersect + n * EPS;
    }

    pathSegment.ray.direction = glm::normalize(wi);
    pathSegment.remainingBounces--;
#if enableRussianRoulette
    if (pathSegment.remainingBounces > 2) {

        float t = fmaxf(pathSegment.color.x, fmaxf(pathSegment.color.y, pathSegment.color.z));
        float rrProb = fminf(1.f, t); 

        if (rrProb < 1.f) {
            if (u01(rng) > rrProb) {
                pathSegment.color = glm::vec3(0.f);
                pathSegment.remainingBounces = 0;
                return;
            }
            pathSegment.color /= rrProb;
        }
    }
#endif
}

