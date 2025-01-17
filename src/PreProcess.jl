module PreProcess

export LoadParticlesFromCSV

using CSV
using DataFrames
using StaticArrays

# This function loads in a CSV file of particles. Please note that it is simply a Vector{SVector{3,Float64}}
# so you can definely make your own! It was just much easier to use a particle distribution layout generated by
# for example DualSPHysics, than making code for this example to do that
function LoadParticlesFromCSV(float_type, fluid_csv,boundary_csv)
    DF_FLUID = CSV.read(fluid_csv, DataFrame)
    DF_BOUND = CSV.read(boundary_csv, DataFrame)

    P1F = DF_FLUID[!,"Points:0"]
    P2F = DF_FLUID[!,"Points:1"]
    P3F = DF_FLUID[!,"Points:2"]
    P1B = DF_BOUND[!,"Points:0"]
    P2B = DF_BOUND[!,"Points:1"]
    P3B = DF_BOUND[!,"Points:2"]

    points           = Vector{SVector{3,float_type}}()
    density_fluid    = Vector{float_type}()
    density_bound    = Vector{float_type}()

    # Since the particles are produced in DualSPHysics
    for i = 1:length(P1F)
        push!(points,SVector(P1F[i],P3F[i],P2F[i]))
        push!(density_fluid,DF_FLUID.Rhop[i])
    end

    for i = 1:length(P1B)
        push!(points,SVector(P1B[i],P3B[i],P2B[i]))
        push!(density_bound,DF_BOUND.Rhop[i])
    end

    return points,density_fluid,density_bound
end

end