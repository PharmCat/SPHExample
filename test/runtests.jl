using Test
using SPHExample
using CSV
using DataFrames
using Printf
using StaticArrays
using CellListMap
import CellListMap: update!
using LinearAlgebra
import ProgressMeter: @showprogress, Progress, next!, BarGlyphs

path    = dirname(@__FILE__)


@testset "  Basic test                                               " begin

function RunSimulation(; 
                        SaveLocation=joinpath(path, "res"),
                        SimulationName="DamBreak",
                        NumberOfIterations=50, # 2 iteration for test 
                        OutputIteration=50)

    progr = Progress(NumberOfIterations,
        dt=1.0,
        barglyphs=BarGlyphs('|','█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
        showspeed=true
    )
    # In the standard folder, we clear results before rerunning simulation
    # foreach(rm, filter(endswith(".vtp"), readdir(SaveLocation,join=true)))

    ### VARIABLE EXPLANATION
    # FLUID_CSV = PATH TO FLUID PARTICLES, SEE "input" FOLDER
    # BOUND_CSV = PATH TO BOUNDARY PARTICLES, SEE "input" FOLDER
    # ρ₀  = REFERENCE DENSITY
    # dx  = INITIAL PARTICLE DISTANCE, SEE "dp" IN CSV FILES, FOR 3D SIM: 0.0085
    # H   = SMOOTHING LENGTH
    # m₀  = INITIAL MASS (REFERENCE DENSITY * DX^(SIMULATION DIMENSIONS))
    # mᵢ  = mⱼ = m₀ | ALL PARTICLES HAVE THE SAME MASS, ALWAYS
    # αD  = NORMALIZATION CONSTANT FOR KERNEL
    # α   = ARTIFICIAL VISCOSITY ALPHA VALUE
    # g   = GRAVITY (POSITIVE!)
    # c₀  = SPEED OF SOUND, MUST BE 10X HIGHEST VELOCITY IN SIMULATION
    # γ   = GAMMA, MOST COMMONLY 7 FOR WATER, USE FOR PRESSURE EQUATION OF STATE
    # dt  = INITIAL TIME STEP
    # δᵩ  = 0.1 | COEFFICIENT FOR DENSITY DIFFUSION, SHOULD ALWAYS BE 0.1
    # CFL = CFL NUMBER

    ### 3D Dam Break - Very slow to solve!
    # FLUID_CSV = "./input/3D_DamBreak_Fluid_3LAYERS.csv"
    # BOUND_CSV = "./input/3D_DamBreak_Boundary_3LAYERS.csv"
    # ρ₀  = 1000
    # dx  = 0.0085
    # H   = sqrt(3)*dx
    # m₀  = ρ₀*dx*dx*dx
    # αD  = 21/(16*π*H^3)
    # α   = 0.01
    # g   = 9.81
    # c₀  = sqrt(g*0.3)*20
    # γ   = 7
    # dt  = 1e-5
    # δᵩ  = 0.1
    # CFL = 0.2

    ### 2D Dam Break
    FLUID_CSV = joinpath(path, "../input/FluidPoints_Dp0.02.csv")
    BOUND_CSV = joinpath(path, "../input/BoundaryPoints_Dp0.02.csv")
    ρ₀  = 1000
    dx  = 0.02
    H   = 1.2 * sqrt(2) * dx
    m₀  = ρ₀ * dx * dx #mᵢ  = mⱼ = m₀
    αD  = (7/(4 * π * H^2))
    α   = 0.01
    g   = 9.81
    c₀  = sqrt(g * 2) * 20
    γ   = 7
    dt  = 1e-5
    δᵩ  = 0.1
    CFL = 0.2

    # Load in the fluid and boundary particles. Return these points and both data frames
    points, DF_FLUID, DF_BOUND    = LoadParticlesFromCSV(FLUID_CSV,BOUND_CSV)

    # Read this as "GravityFactor * g", so -1 means negative acceleration for fluid particles
    # 1 means boundary particles push back against gravity
    GravityFactor = [-ones(size(DF_FLUID,1)) ; ones(size(DF_BOUND,1))]

    # MotionLimiter is what allows fluid particles to move, while not letting the velocity of boundary
    # particles change
    MotionLimiter = [ ones(size(DF_FLUID,1)) ; zeros(size(DF_BOUND,1))]

    # Based on MotionLimiter we assess which particles are boundary particles
    BoundaryBool  = .!Bool.(MotionLimiter)


    # Initialize arrays
    density      = Array([DF_FLUID.Rhop;DF_BOUND.Rhop])
    velocity     = zeros(eltype(points),length(points))
    acceleration = zeros(eltype(points),length(points))

    # Save the initial particle layout with dummy values
    # Disable for test
    # create_vtp_file(SaveLocation*"/"*SimulationName*"_"*lpad("0",4,"0"),points,density.*0,acceleration.*0,density,Pressure.(density,c₀,γ,ρ₀),acceleration,velocity)

    # Initialize the system list
    system  = InPlaceNeighborList(x=points, cutoff= 2H, parallel=true)
    
    N    = length(points)

    viscI = zeros(SVector{3, Float64}, N)
    dvdtI = zeros(SVector{3, Float64}, N)
    dρdtI = zeros(N)

    xᵢⱼ   = zeros(SVector{3, Float64}, system.nb.n)

    WgI = zeros(SVector{3, Float64}, N)
    WgL = zeros(SVector{3, Float64}, system.nb.n)

    viscL = zeros(SVector{3, Float64}, system.nb.n)
    dvdtL = zeros(SVector{3, Float64}, system.nb.n)
    dρdtL = zeros(system.nb.n)

    drhopvpbuffer  = zeros(Float64, system.nb.n)
    drhopvnbuffer  = zeros(Float64, system.nb.n)

    density_n_half  = zeros(N)
    velocity_n_half = zeros(SVector{3, Float64}, N)
    points_n_half   = zeros(SVector{3, Float64}, N)

    epsi            = zeros(N)

    density_new     = zeros(N)
    velocity_new    = zeros(SVector{3, Float64}, N)
    points_new      = zeros(SVector{3, Float64}, N)
    
    for sim_iter = 1:NumberOfIterations
        # Be sure to update and retrieve the updated neighbour list at each time step
        update!(system, points)
        list = neighborlist!(system)
        
        updatexᵢⱼ!(xᵢⱼ, list, points)

        resizebuffers!(WgL, viscL, dvdtL, dρdtL, drhopvpbuffer, drhopvnbuffer; N = system.nb.n)

        # Here we output the kernel value for each particle
        WiI, _   = ∑ⱼWᵢⱼ(list, points, αD, H)
        # Here we output the kernel gradient value for each particle and also the kernel gradient value
        # based on the pair-to-pair interaction list, for use in later calculations.
        # Other functions follow a similar format, with the "I" and "L" ending
        
        WgI, WgL = ∑ⱼ∇ᵢWᵢⱼ!(WgI, WgL, xᵢⱼ, list, points, αD, H) 

        # Then we calculate the density derivative at time step "n"
        dρdtI, _ = ∂ρᵢ∂tDDT!(dρdtI, dρdtL, xᵢⱼ, list, points, H, m₀, δᵩ, c₀, γ, g, ρ₀, density, velocity, WgL, MotionLimiter; drhopvp = drhopvpbuffer, drhopvn = drhopvnbuffer)

        # We calculate viscosity contribution and momentum equation at time step "n"
        viscI, _ = ∂Πᵢⱼ∂t!(viscI, viscL, xᵢⱼ, list, points, H, density, α, velocity, c₀, m₀, WgL)

        dvdtI, _ = ∂vᵢ∂t!(dvdtI, dvdtL, list, points, m₀, density, WgL, c₀, γ, ρ₀)
        # We add gravity as a final step for the i particles, not the L ones, since we do not split the contribution, that is unphysical!
        # So please be careful with using "L" results directly in some cases
        dvdtI .= map((x,y) -> x + y * SVector(0, g, 0), dvdtI + viscI, GravityFactor)


        # Based on the density derivative at "n", we calculate "n+½"
        @. density_n_half  = density + dρdtI * (dt/2)
        # We make sure to limit the density of boundary particles in such a way that they cannot produce suction
        density_n_half[(density_n_half .< ρ₀) .* BoundaryBool] .= ρ₀

        # We now calculate velocity and position at "n+½"
        @. velocity_n_half = velocity + dvdtI * (dt/2) * MotionLimiter
        @. points_n_half   =  points   + velocity_n_half * (dt/2) * MotionLimiter

        updatexᵢⱼ!(xᵢⱼ, list, points_n_half)

        # Density derivative at "n+½" - Note that we keep the kernel gradient values calculated at "n" for simplicity
        dρdtI_n_half, _ = ∂ρᵢ∂tDDT!(dρdtI, dρdtL, xᵢⱼ, list, points_n_half, H, m₀, δᵩ, c₀, γ, g, ρ₀, density_n_half, velocity_n_half, WgL, MotionLimiter; drhopvp = drhopvpbuffer, drhopvn = drhopvnbuffer)
        # Viscous contribution and momentum equation at "n+½"

        viscI_n_half,_ = ∂Πᵢⱼ∂t!(viscI, viscL, xᵢⱼ, list, points_n_half, H, density_n_half, α, velocity_n_half, c₀, m₀, WgL)

        dvdtI_n_half,_ = ∂vᵢ∂t!(dvdtI, dvdtL, list, points_n_half, m₀, density_n_half, WgL, c₀, γ, ρ₀)
        dvdtI_n_half  .= map((x, y)->x + y * SVector(0, g, 0), dvdtI_n_half + viscI_n_half, GravityFactor) 

        # Factor for properly time stepping the density to "n+1" - We use the symplectic scheme as done in DualSPHysics
        @. epsi = -(dρdtI_n_half / density_n_half) * dt

        # Finally we update all values to their next time step, "n+1"
        @. density_new   =  density  * (2 - epsi)/(2 + epsi)
        density_new[(density_new .< ρ₀) .* BoundaryBool] .= ρ₀
        @. velocity_new  = velocity + dvdtI_n_half * dt * MotionLimiter
        @. points_new    =  points   + ((velocity_new + velocity)/2) * dt * MotionLimiter

        # And for clarity updating the values in our simulation is done explicitly here
        density      .= density_new
        velocity     .= velocity_new
        points       .= points_new
        acceleration .= dvdtI_n_half

        # Automatic time stepping control
        dt = Δt(acceleration, points, velocity, c₀, H, CFL)

        #@printf "Iteration %i | dt = %.5e \n" sim_iter dt
        # Disable for test
        #if sim_iter % OutputIteration == 0
        #    create_vtp_file(SaveLocation*"/"*SimulationName*"_"*lpad(sim_iter,4,"0"),points,WiI,WgI,density,Pressure.(density,c₀,γ,ρ₀),acceleration,velocity)
        #end
        next!(progr; showvalues = [(:iter, sim_iter), (:dt, dt)])
    end
    points, density, pressure.(density, c₀, γ, ρ₀), acceleration, velocity
end

# Test result
points, density, pres, acceleration, velocity = RunSimulation(NumberOfIterations=2)

    # Dev test, may not reflect real correctness

    @test sum(points) ≈ [5475.120088053588
    6215.819913178543
       0.0]

    @test sum(density) ≈ 6.200838532554191e6 atol = 6.200838532554191e6 * 1E-5

    @test sum(pres) ≈ 4.604937728177776e7 atol = 4.604937728177776e7 * 1E-2

    @test sum(acceleration) ≈ [ 1.4551915228366852e-11, -31519.530000000093, 0.0] atol = maximum(abs.([ 1.4551915228366852e-11, -31519.530000000093, 0.0])) * 1E-6

    @test sum(velocity) ≈ [2.0295953409284935, -2.0079407428925227, 0.0]

end

#=
using Profile
using PProf

@profile  for i = 1:3  RunSimulation() end

pprof()
#PProf.kill()
#Profile.clear()
=#

#=
using BenchmarkTools
@btime RunSimulation(NumberOfIterations=200);
=#


#=
function btpn(cpupoints, dist)
    n = 0
    for i = 1:length(cpupoints)-1
        for j = i+1:length(cpupoints)
            if sqrt((cpupoints[i][1] -  cpupoints[j][1])^2 + (cpupoints[i][2] -  cpupoints[j][2])^2) < dist 
                n += 1 
            end
        end
    end
    n
end
btpn(points, 2H)
=#