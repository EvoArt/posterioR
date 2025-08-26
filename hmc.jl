using LinearAlgebra, GLMakie, Distributions

update_momentum(p,q,ϵ,∇U) = p .- (ϵ/2) * ∇U(q)
update_position(p,q,ϵ,M⁻¹) = q .+ ϵ .* M⁻¹ * p
function update_pq!(p,q,ϵ,∇U,M⁻¹)
    p .= update_momentum(p,q,ϵ,∇U) 
    q .= update_position(p,q,ϵ,M⁻¹)
    p .= update_momentum(p,q,ϵ,∇U) 
end


U(x,μ=1.0,σ =2.0) = -log(σ) + 0.5log(2π) + 0.5((x-μ)/σ)^2
U(x::Array) = sum(U,x)

∇U(x,μ=1.0,σ =2.0) = (x - μ) / σ^2
∇U(x::Array) = ∇U.(x)

function hmc_step(q,ϵ,L,M,M⁻¹,U,∇U,H,p_dist = MvNormal(M))
    q_proposal = copy(q)
    p = rand(p_dist) 
    H_curr = U(q) + 0.5 * dot(p, M⁻¹ * p)
    for l in 1:L
        update_pq!(p,q_proposal,ϵ,∇U,M⁻¹)
    end
    H_proposal =U(q_proposal) + 0.5 * dot(p, M⁻¹ * p)
    prob = min(1, exp(H_curr - H_proposal))

    accept = prob >= rand()

    if accept
        return q_proposal,H_proposal 
    else
        return q,H_curr 
    end
end
function hmc(n_iter = 1000,n = 1,ϵ = 0.1,L = 40,M = I(n),M⁻¹ = M^-1,U=U,∇U=∇U,p_dist = MvNormal(M))
    q = randn(n)
    qs = []
    q,H = hmc_step(q,ϵ,L,M,M⁻¹,U,∇U,0.0)
    for i in 1:n_iter
        q,H =  hmc_step(q,ϵ,L,M,M⁻¹,U,∇U,H)
        push!(qs,q...)
    end
    return qs
end

samples =  hmc(10000)
μ,σ = round.([mean(samples),std(samples)],sigdigits =3)
fig = Figure()
ax = Axis(fig[1,1], title = "trace")
lines!(ax,samples)
ax = Axis(fig[1,2], title = "density: μ = $μ, σ = $σ")
density!(ax,samples)


