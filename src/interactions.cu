#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>
#include <glm/gtx/norm.hpp>

#define enableRussianRoulette 0

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

__host__ __device__ inline float sampleRadiusTwoExp(thrust::default_random_engine& rng,
    float sigma_t_avg,
    float scale,
    float minSigma, 
    float w1,   
    float slowMul,      
    float fastMul,      
    float& pdf)     
{
    thrust::uniform_real_distribution<float> U(0.f, 1.f);


    float uComp = U(rng);
    float w2 = 1.f - w1;

    float s1 = fmaxf(slowMul * sigma_t_avg, minSigma);
    float s2 = fmaxf(fastMul * sigma_t_avg, minSigma);

    float u = U(rng);
    float r = (uComp < w1)
        ? -logf(fmaxf(1.f - u, 1e-6f)) / s1
        : -logf(fmaxf(1.f - u, 1e-6f)) / s2;

    r *= scale;

    float rBase = r / scale;
    float pdf_base = w1 * s1 * expf(-s1 * rBase) + w2 * s2 * expf(-s2 * rBase);
    pdf = fmaxf(pdf_base / scale, 1e-6f);

    return r;
}

__host__ __device__ inline glm::vec3 sampleHGWorld(
    thrust::default_random_engine& rng,
    float g,
    const glm::vec3& axisN ) 
{
    thrust::uniform_real_distribution<float> U(0.f, 1.f);
    float u1 = U(rng), u2 = U(rng);

    float cosTheta;
    if (fabsf(g) < 1e-4f) {
        cosTheta = 1.f - 2.f * u1;
    }
    else {
        float s = (1.f - g * g) / (1.f - g + 2.f * g * u1);
        cosTheta = (1.f + g * g - s * s) / (2.f * g);
        cosTheta = fminf(fmaxf(cosTheta, -1.f), 1.f);
    }
    float sinTheta = sqrtf(fmaxf(0.f, 1.f - cosTheta * cosTheta));
    float psi = TWO_PI * u2;

    glm::vec3 local(sinTheta * cosf(psi), sinTheta * sinf(psi), cosTheta);

    glm::vec3 N = glm::normalize(axisN);
    glm::vec3 T = (fabsf(N.x) < 0.9f) ? glm::normalize(glm::cross(N, glm::vec3(1, 0, 0)))
        : glm::normalize(glm::cross(N, glm::vec3(0, 1, 0)));
    glm::vec3 B = glm::normalize(glm::cross(N, T));

    return glm::normalize(local.x * T + local.y * B + local.z * N);
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

    if (m.hasSubsurface > 0.f) {
        
        const float edgeBoost      = 1.0f;     
        const float energyClamp    = 8.0f;  
        const float mixLambertFrac = 0.15f;
        const float maxRadiusMul   = 4.0f;
        const float gPhase         = m.hgG;
        const float minSigma       = 1e-6f;

        glm::vec3 sigma_s = m.sigma_s;
        glm::vec3 sigma_a = m.sigma_a;
        glm::vec3 sigma_t = sigma_s + sigma_a;
        glm::vec3 safe_sigma_t = glm::max(sigma_t, glm::vec3(minSigma));
        float sigma_t_avg = (safe_sigma_t.x + safe_sigma_t.y + safe_sigma_t.z) / 3.f;

        glm::vec3 N = n;
        glm::vec3 T = (fabsf(N.x) < 0.9f) ?
            glm::normalize(glm::cross(N, glm::vec3(1,0,0))) :
            glm::normalize(glm::cross(N, glm::vec3(0,1,0)));
        glm::vec3 B = glm::normalize(glm::cross(N, T));


        float pdfR;
        float r = sampleRadiusTwoExp(rng,
            sigma_t_avg,
            m.mediumScale,
            1e-6f,
            /*w1=*/0.65f,
            /*slowMul=*/0.5f,
            /*fastMul=*/2.0f,
            pdfR);
        float rMax = maxRadiusMul * m.mediumScale;
        if (r > rMax) {
            r = rMax;
            pdfR = fmaxf(pdfR, 1.f / (rMax + 1e-3f));
        }

        thrust::uniform_real_distribution<float> u01(0.f,1.f);
        float phi = TWO_PI * u01(rng);
        glm::vec3 lateral = T * (r * cosf(phi)) + B * (r * sinf(phi));

        glm::vec3 atten = glm::vec3(
            expf(-sigma_a.x * r),
            expf(-sigma_a.y * r),
            expf(-sigma_a.z * r)
        );
        glm::vec3 albedo = sigma_s / safe_sigma_t;

        float viewGrazing = 1.f - fmaxf(0.f, fabsf(glm::dot(N, -inVec)));
        float rim = powf(fmaxf(viewGrazing, 0.f), 1.5f) * edgeBoost;

        glm::vec3 weight = (albedo * atten * m.color) * (rim + 1.f);
        weight /= pdfR;
        weight = glm::min(weight, glm::vec3(energyClamp));

        glm::vec3 outDir = sampleHGWorld(rng, m.hgG, N);
        if (glm::dot(outDir, N) < 0.f) outDir = -outDir;

        glm::vec3 lambert = calculateRandomDirectionInHemisphere(N, rng);
        if (glm::dot(lambert, N) < 0.f) lambert = -lambert;
        outDir = glm::normalize((1.f - mixLambertFrac) * outDir + mixLambertFrac * lambert);

        glm::vec3 exitPoint = intersect + lateral + N * 1e-4f;

        pathSegment.color *= weight;
        pathSegment.ray.origin = exitPoint;
        pathSegment.ray.direction = outDir;
        pathSegment.remainingBounces--;
        return;
    }

    if (m.hasRefractive > 0.f && m.indexOfRefraction > 0.f)
    {
        if(m.flip > 0.f)
			n = -n;
        bool in_or_out = glm::dot(inVec, n) < 0.f;
        glm::vec3 nDir = in_or_out ? n : -n;

        float etaI = in_or_out ? 1.f : m.indexOfRefraction;
        float etaT = in_or_out ? m.indexOfRefraction : 1.f;
        float eta_ratio = etaI / etaT; 

        float consine = fminf(fmaxf(-glm::dot(inVec, nDir), 0.f), 1.f);
        float sin2 = eta_ratio * eta_ratio * fmaxf(0.f, 1.f - consine * consine);
		bool cannotRefract = sin2 > 1.f;

        /*float F = reflectance(consine, etaI, etaT);*/
        float cosT = sqrtf(fmaxf(0.f, 1.f - sin2));
        float F = reflectance(in_or_out ? consine : cosT, etaI, etaT);

        glm::vec3 newDir;
        float rand_f = u01(rng);
        if (m.isThin > 0.f) 
        {
            float cosI = glm::clamp(-dot(inVec, nDir), 0.f, 1.f);
            float F = reflectance(cosI, 1.f, m.indexOfRefraction);

            if (u01(rng) < F) {
               
                newDir = reflect(inVec, nDir);
                pathSegment.color *= glm::vec3(1.f);
                pathSegment.ray.origin = intersect + nDir * EPS;   
            }
            else {
                
                newDir = inVec;
                pathSegment.color *= glm::vec3(1.f);                    
                pathSegment.ray.origin = intersect + newDir * EPS;  

            }

            pathSegment.ray.direction = normalize(newDir);
            pathSegment.remainingBounces--;
            return;
        }
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

