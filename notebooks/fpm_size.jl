### A Pluto.jl notebook ###
# v0.18.1

using Markdown
using InteractiveUtils

# ╔═╡ b487b43a-0785-11ec-37a0-d7a8c2311d96
begin
    using Markdown
    using Unitful
    using UnitfulAngles
end

# ╔═╡ c62c6afa-85c2-4699-a1c2-ee815680f359
md"""
## Derivation of focal plane mask sizes

### Optics of Subaru/VAMPIRES
"""

# ╔═╡ 9f7970f3-abe7-4c75-9240-fd00dee8d7a5
subaru_diameter = 8.2u"m"

# ╔═╡ 7f22f26b-aae3-4684-b01c-711567ec9e09
clear_aperture = subaru_diameter * 0.95

# ╔═╡ 82befe1c-06b8-4aff-af57-b54d204f6cda
# measured from pupil images of aperture masks
# which have known sizes
effective_diameter = 3.886779904306219u"mm" * 2

# ╔═╡ 69ec058c-a65d-45da-b73a-7c93352775af
focal_plane_coll_length = 200u"mm"

# ╔═╡ 6257f73a-daf1-4acb-a3ca-fcf2da357ea6
F_ratio = focal_plane_coll_length / effective_diameter |> u"rad"

# ╔═╡ b44f0439-3e83-4039-b9d4-c0b80137f723
effective_focal_length = clear_aperture * F_ratio |> u"m"

# ╔═╡ c6697852-24bb-4a73-b2b8-741c5460db96
effective_plate_scale = inv(effective_focal_length) |> u"arcsecond/mm"

# ╔═╡ 057f8478-3ac9-4b26-b1f9-e6706841cc7b
md"""
### Focal Plane Masks

For the given wavelengths, find the linear size of the FPM mask diameter, given the number of ``\lambda/D`` radius.

```math
d_\mathrm{FPM} = \frac{2k\cdot\lambda/D}{\bar{p}}
```
"""

# ╔═╡ c2507457-7115-4cd2-a03f-dff0eb8fe67b
target_wavelength = 750u"nm"

# ╔═╡ 8e9e0d9e-c91e-4830-8637-8a795debe184
fpm_masks = [2, 3, 5, 7] # lambda / D, radius

# ╔═╡ e0f88a2f-eed1-4200-accb-81739068d2ab
# first column in 656nm, second is 750nm
fpm_masks_angle = @. fpm_masks * target_wavelength / clear_aperture |> u"arcsecond"

# ╔═╡ 9cf95d35-6a17-4752-99dc-f4580e1b3c0e
fpm_masks_radius = @. fpm_masks_angle / effective_plate_scale |> u"μm"

# ╔═╡ 7b5be534-da03-45ba-a2c6-1f1b7fdbc917
fpm_masks_angle_calculated = @. fpm_masks_radius /
                                (clear_aperture * focal_plane_coll_length /
                                 (2 * 3.1644u"mm" / 0.9)) |> u"arcsecond" |> x -> x * 1e3

# ╔═╡ c5a7185b-34f5-4aaa-87dc-f687c5785c47
fpm_masks_diameter = 2 .* fpm_masks_radius

# ╔═╡ 1fb30e04-dba0-4d2b-8151-2c64cae46592
md"### Beam shift measurement"

# ╔═╡ df4c6716-02ba-4131-ba7f-83cda9ea6688
# https://www.edmundoptics.com/p/25mm-dia-3mm-thick-nir-i-coated-lambda10-fused-silica-window/27561/
begin
    n = 1.458
    α = 5u"arcsecond"
end;

# ╔═╡ 779111b8-e4c9-4ba5-9d41-4b8829d9b29f
# beam shift
δ = 2 * (n - 1) * α

# ╔═╡ 9bbf3483-050b-4d5d-b49f-3f533417393c
δ / (target_wavelength / effective_diameter) |> u"NoUnits"

# ╔═╡ 172bba75-bd99-4f9e-953b-d92a7ebcb5fd
md"""
---
**imports**
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Markdown = "d6f4376e-aef5-505a-96c1-9c027394607a"
Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"
UnitfulAngles = "6fb2a4bd-7999-5318-a3b2-8ad61056cd98"

[compat]
Unitful = "~1.9.0"
UnitfulAngles = "~0.6.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.0+0"

[[ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f74e9d5388b8620b4cee35d4c5a618dd4dc547f4"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.3.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.17+2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Unitful]]
deps = ["ConstructionBase", "Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "a981a8ef8714cba2fd9780b22fd7a469e7aaf56d"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.9.0"

[[UnitfulAngles]]
deps = ["Dates", "Unitful"]
git-tree-sha1 = "dd21b5420bf6e9b76a8c6e56fb575319e7b1f895"
uuid = "6fb2a4bd-7999-5318-a3b2-8ad61056cd98"
version = "0.6.1"

[[libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "4.0.0+0"
"""

# ╔═╡ Cell order:
# ╟─c62c6afa-85c2-4699-a1c2-ee815680f359
# ╟─9f7970f3-abe7-4c75-9240-fd00dee8d7a5
# ╟─7f22f26b-aae3-4684-b01c-711567ec9e09
# ╠═82befe1c-06b8-4aff-af57-b54d204f6cda
# ╟─69ec058c-a65d-45da-b73a-7c93352775af
# ╠═6257f73a-daf1-4acb-a3ca-fcf2da357ea6
# ╠═b44f0439-3e83-4039-b9d4-c0b80137f723
# ╠═c6697852-24bb-4a73-b2b8-741c5460db96
# ╟─057f8478-3ac9-4b26-b1f9-e6706841cc7b
# ╠═c2507457-7115-4cd2-a03f-dff0eb8fe67b
# ╟─8e9e0d9e-c91e-4830-8637-8a795debe184
# ╠═e0f88a2f-eed1-4200-accb-81739068d2ab
# ╠═9cf95d35-6a17-4752-99dc-f4580e1b3c0e
# ╠═7b5be534-da03-45ba-a2c6-1f1b7fdbc917
# ╠═c5a7185b-34f5-4aaa-87dc-f687c5785c47
# ╟─1fb30e04-dba0-4d2b-8151-2c64cae46592
# ╠═df4c6716-02ba-4131-ba7f-83cda9ea6688
# ╠═779111b8-e4c9-4ba5-9d41-4b8829d9b29f
# ╠═9bbf3483-050b-4d5d-b49f-3f533417393c
# ╟─172bba75-bd99-4f9e-953b-d92a7ebcb5fd
# ╠═b487b43a-0785-11ec-37a0-d7a8c2311d96
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
