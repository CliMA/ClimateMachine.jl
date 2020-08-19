struct MetricTerms
    ∂x
    ∂y
    ∂z
end

struct VolumeGeometry
    coordinates
    jacobian
    metricterms
    mass 
    inversemass
end

struct LocalCommunication
    minus
    plus
end

struct MPICommunication
    interior
    exterior
end

struct SurfaceGeometry
    coordinates
    mass
    inversemass
end

struct ElementOperators
    mass
    inversemass
    DifferentiationMatrix
end

