### A Pluto.jl notebook ###
# v0.18.1

using Markdown
using InteractiveUtils

# ╔═╡ d96f4a03-6bab-4569-9b05-950469b74648
begin
	using Pkg
	Pkg.activate("..")
end

# ╔═╡ eade1b24-0608-11ec-2c93-abfde494572f
begin
	using Dierckx
	using Markdown
	using Plots
	using Printf
	using Statistics
	using Unitful
	using Unitful: c0, μ0
	using UnitfulRecipes
end

# ╔═╡ 2cdfd99e-c42f-4db6-9287-efd531959ca3
md"""
## Skin Depth Calculations

The penetration depth is the distance over which light intensity decreases by a factor of ``e``. The penetration depth can be calculated from the complex index of refraction-

```math
n(\lambda) = \tilde{n}(\lambda) - i \tilde{k}(\lambda)
```

To calculate the penetration depth we use the formula 

```math
\delta_p(\lambda) \equiv \frac{1}{\alpha(\lambda)} = \frac{\lambda}{4\pi\tilde{k}(\lambda)}
```

Now, we can calculate the amount of attenuation-

```math
\frac{I(s, \lambda)}{I_0(\lambda)} =  e^{-\alpha(\lambda)s}
```

or likewise, the thickness required for a given attenuation

```math
\hat{s}(\lambda) = -\delta_p(\lambda) \log{\frac{I}{I_0}}
```
"""

# ╔═╡ 3e67833e-ceff-42a1-afb1-7670dabd2fc0
penetration_depth(λ, k) = λ / (4π * k)

# ╔═╡ 4788fc3c-b833-4d3a-9aae-8168cd0de80f
thickness(attenuation, λ, k) = -penetration_depth(λ, k) * log(attenuation)

# ╔═╡ b92fc0c0-fbeb-46ad-b17f-cdd5c7094eaf
attenuation(thickness, λ, k) = exp(-thickness / penetration_depth(λ, k))

# ╔═╡ af3c388b-4297-4542-9202-ed467b72aadb
λ = range(600, 800, length=100)u"nm"

# ╔═╡ c702f78b-3cf0-4fd3-ba6b-ee41beb57c2c
md"""
Getting all my values from [here](http://www.angstec.com/graph/24)
"""

# ╔═╡ c8f5ed5e-50c5-4b9a-98f8-997ac5841b5a
Au = let tbl, itp
	tbl = [0.5905	0.2360	2.9700
		   0.6049	0.2124	3.0150
		   0.6200	0.1940	3.0600
   		   0.6359	0.1778	3.0700
		   0.6526	0.1660	3.1500
		   0.6703	0.1610	3.4458
		   0.6889	0.1600	3.8000
		   0.7086	0.1609	4.0877
		   0.7294	0.1640	4.3570
		   0.7515	0.1695	4.6102
		   0.7750	0.1760	4.8600
		   0.8000	0.1814	5.1258
		   0.8267	0.1880	5.3900]
	itp = Spline1D(tbl[:, 1], tbl[:, 3])
	itp.(ustrip.(u"μm", λ))
end

# ╔═╡ 6d878fb2-9c75-4b4e-8195-875189fca6ad
let thicknesses, trans, ts, mean_trans, mean_thick
	thicknesses = range(0, 500, length=101)u"nm"
	heatmap(
		thicknesses, λ, log10.(attenuation.(thicknesses', λ, Au)),
		c=:inferno,
		xlabel="thickness",
		ylabel="λ",
		title="Gold (Au)",
		cbtitle="log10(transmission)",
		leg=:topleft,
		clim=(-15, 0)
	)
	trans = 1e-8
	ts = @. thickness(trans, λ, Au) |> u"nm"
	plot!(
		ts, λ,
		label=@sprintf("%.1e", trans),
		c=:black,
		lw=2,
		xlim=extrema(thicknesses),
		ylim=extrema(λ)
	)
	mean_thick = mean(ts)
	std_thick = std(ts, mean=mean_thick)
	lab = "$(round(u"nm", mean_thick)) ± $(round(u"nm", std_thick))" 
	vline!([mean_thick],
		c=:black,
		ls=:dash,
		label=lab
	)
end

# ╔═╡ 5d0b199e-c985-42fc-aca7-9f4fb53df668
let thicknesses, trans, ts, mean_trans, mean_thick
	thicknesses = range(0, 500, length=101)u"nm"
	trans_au = @. 0.5 * attenuation(thicknesses', λ, Au)
	trans_cr = @. 0.5 * attenuation(thicknesses', λ, Cr)
	heatmap(
		thicknesses, λ, log10.(trans_au .+ trans_cr),
		c=:inferno,
		xlabel="thickness",
		ylabel="λ",
		title="1/2 Gold (Au) + 1/2 Chromium (Cr)",
		cbtitle="log10(transmission)",
		leg=:topleft,
		clim=(-15, 0)
	)
	trans = 1e-8
	ts_au = @. 0.5 * thickness(trans, λ, Au) |> u"nm"
	ts_cr = @. 0.5 * thickness(trans, λ, Cr) |> u"nm"
	ts = ts_au .+ ts_cr
	plot!(
		ts, λ,
		label=@sprintf("%.1e", trans),
		c=:black,
		lw=2,
		xlim=extrema(thicknesses),
		ylim=extrema(λ)
	)
	mean_thick = mean(ts)
	std_thick = std(ts, mean=mean_thick)
	lab = "$(round(u"nm", mean_thick)) ± $(round(u"nm", std_thick))" 
	vline!([mean_thick],
		c=:black,
		ls=:dash,
		label=lab
	)
end

# ╔═╡ 1bfd39a9-f3aa-41d6-b188-e45f1067cb5d
Cr = let tbl, itp
	tbl = [0.5905	3.2100	3.3000
		   0.6049	3.1881	3.2969
	       0.6200	3.1600	3.3000
		   0.6359	3.1294	3.3125
		   0.6526	3.1000	3.3300
		   0.6703	3.0763	3.3500
		   0.6889	3.0600	3.3700
		   0.7086	3.0544	3.3850
		   0.7294	3.0600	3.4000
		   0.7515	3.0787	3.4194
		   0.7750	3.1100	3.4400
		   0.8000	3.1581	3.4600
		   0.8267	3.2100	3.4800]
	itp = Spline1D(tbl[:, 1], tbl[:, 3])
	itp.(ustrip.(u"μm", λ))
end

# ╔═╡ 5b3858e6-abea-49cc-978b-af239f26188e
penetration_depth(700u"nm", mean(Cr)) |> u"nm"

# ╔═╡ f4bf5b37-d983-451e-ae83-c0736365aeee
let thicknesses, trans, ts, mean_trans, mean_thick
	thicknesses = range(0, 500, length=101)u"nm"
	heatmap(
		thicknesses, λ, log10.(attenuation.(thicknesses', λ, Cr)),
		c=:viridis,
		xlabel="thickness",
		ylabel="λ",
		title="Chromium (Cr)",
		cbtitle="log10(transmission)",
		xlim=extrema(thicknesses),
		ylim=extrema(λ),
		leg=:topleft
	)
	trans = 1e-3
	ts = @. thickness(trans, λ, Cr) |> u"nm"
	plot!(
		ts, λ,
		label=@sprintf("%.1e", trans),
		c=:black,
		alpha=0.4,
		lw=2,
	)
	mean_thick = mean(ts)
	std_thick = std(ts, mean=mean_thick)
	lab = "$(round(u"nm", mean_thick)) ± $(round(u"nm", std_thick))"
	vline!([mean_thick],
		c=:black,
		alpha=0.4,
		ls=:dash,
		label=lab
	)
	
	trans = 1e-8
	ts = @. thickness(trans, λ, Cr) |> u"nm"
	plot!(
		ts, λ,
		label=@sprintf("%.1e", trans),
		c=:black,
		lw=2,
	)
	mean_thick = mean(ts)
	std_thick = std(ts, mean=mean_thick)
	lab = "$(round(u"nm", mean_thick)) ± $(round(u"nm", std_thick))"
	vline!([mean_thick],
		c=:black,
		ls=:dash,
		label=lab
	)
end

# ╔═╡ 45aed391-b9d5-4289-ad01-b4494db0a76b
md"""
---

**Imports**
"""

# ╔═╡ Cell order:
# ╟─2cdfd99e-c42f-4db6-9287-efd531959ca3
# ╠═3e67833e-ceff-42a1-afb1-7670dabd2fc0
# ╠═4788fc3c-b833-4d3a-9aae-8168cd0de80f
# ╠═b92fc0c0-fbeb-46ad-b17f-cdd5c7094eaf
# ╟─af3c388b-4297-4542-9202-ed467b72aadb
# ╟─c702f78b-3cf0-4fd3-ba6b-ee41beb57c2c
# ╟─c8f5ed5e-50c5-4b9a-98f8-997ac5841b5a
# ╟─6d878fb2-9c75-4b4e-8195-875189fca6ad
# ╟─5d0b199e-c985-42fc-aca7-9f4fb53df668
# ╠═5b3858e6-abea-49cc-978b-af239f26188e
# ╟─1bfd39a9-f3aa-41d6-b188-e45f1067cb5d
# ╟─f4bf5b37-d983-451e-ae83-c0736365aeee
# ╟─45aed391-b9d5-4289-ad01-b4494db0a76b
# ╠═d96f4a03-6bab-4569-9b05-950469b74648
# ╠═eade1b24-0608-11ec-2c93-abfde494572f
