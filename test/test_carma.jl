using Pioran, Test, Random

function test_quad2roots32()
	qa32 = [0.025443151049354032,
		0.04252858046335997,
		2.5980088198563633]
	r32 = [-0.021264290231679986 + 0.1580853598860341im,
		-0.021264290231679986 - 0.1580853598860341im,
		-2.5980088198563633 + 0.0im]
	r = quad2roots(qa32)
	@test r ≈ r32
end

function test_roots2coeffs32()
	r32 = [-0.012721575524677016 + 0.20583182936448363im,
		-0.012721575524677016 - 0.20583182936448363im,
		-2.5980088198563633 + 0.0im]
	α = roots2coeffs(r32)
	α32 = [0.11048962713978024,
		0.10863011129451944,
		2.6234519709057174,
		1]
	@test α ≈ α32
end

function test_init_CARMA32()
	rα = [-0.042163209825323775 + 1.1115603157767922im,
		-0.042163209825323775 - 1.1115603157767922im,
		-0.7599101571312047 + 0.0im]
	β = [3.9413022090550216,
		11.38193903188344,
		1]
	𝓒 = CARMA(3, 2, rα, β, 1.3)
	@test 𝓒 isa CARMA
	@test 𝓒.p == 3
	@test 𝓒.q == 2
end

function test_celerite_coefCARMA()
	rα = [-0.042163209825323775 + 1.1115603157767922im,
		-0.042163209825323775 - 1.1115603157767922im,
		-0.7599101571312047 + 0.0im]
	β = [3.9413022090550216,
		11.38193903188344,
		1]
	𝓒 = CARMA(3, 2, rα, β, 1.3)
	a, b, c, d = Pioran.celerite_coefs(𝓒)
	a_, b_, c_, d_ = [1.332733901854476, -0.03273390185447589], [-0.026820976815752837, 0.0], [0.042163209825323775, 0.7599101571312047], [-1.1115603157767922, 0.0]
	@test a ≈ a_
	@test b ≈ b_
	@test c ≈ c_
	@test d ≈ d_
end

function test_celerite_repr()
	rα = [-0.042163209825323775 + 1.1115603157767922im,
		-0.042163209825323775 - 1.1115603157767922im,
		-0.7599101571312047 + 0.0im]
	β = [3.9413022090550216,
		11.38193903188344,
		1]
	𝓒 = CARMA(3, 2, rα, β, 1.3)
	C = Pioran.celerite_repr(𝓒)

	@test C isa Pioran.SumOfSemiSeparable

	a, b, c, d = C.a, C.b, C.c, C.d
	a_, b_, c_, d_ = [1.332733901854476, -0.03273390185447589], [-0.026820976815752837, 0.0], [0.042163209825323775, 0.7599101571312047], [-1.1115603157767922, 0.0]
	@test a ≈ a_
	@test b ≈ b_
	@test c ≈ c_
	@test d ≈ d_

end

function test_CARMA_ACVF()
	t = LinRange(0, 150, 1000)
	rα = [-0.042163209825323775 + 1.1115603157767922im,
		-0.042163209825323775 - 1.1115603157767922im,
		-0.7599101571312047 + 0.0im]
	β = [3.9413022090550216,
		11.38193903188344,
		1]
	𝓒 = CARMA(3, 2, rα, β, 1.3)
	C = Pioran.celerite_repr(𝓒)
	ACVF_cel = C.(t, 0.0)
	ACVF_carma = Pioran.CARMA_covariance(t, 𝓒)
	@test ACVF_cel ≈ ACVF_carma
end

function test_CARMA_PSD()
    f = 10.0 .^ LinRange(-3, 3, 1000)
    rα = [-0.042163209825323775 + 1.1115603157767922im,
		-0.042163209825323775 - 1.1115603157767922im,
		-0.7599101571312047 + 0.0im]
	β = [3.9413022090550216,
		11.38193903188344,
		1]
	𝓒 = CARMA(3, 2, rα, β,1.0)

    a,b,c,d = Pioran.celerite_coefs(𝓒)
    psd_cel = 2*sum([Pioran.Celerite_psd.(f,Ref(a[i]), Ref(b[i]), Ref(c[i]), Ref(d[i])) for i in 1:length(a)]);
    psd_carma = Pioran.calculate(f,𝓒)/Pioran.CARMA_normalisation(𝓒);
	@test psd_cel ≈ psd_carma
end

function test_init_CARMA52()
	rα = [-0.11080991083705849 + 0.0im,
		-1.606382817428408 + 0.0im,
		-0.9960156700480726 + 1.5981072723525709im,
		-0.9960156700480726 - 1.5981072723525709im,
		-0.35149763897615616 + 0.0im]

	β = [0.02335073568393333,
		10.30196190264543,
		1]

	𝓒 = CARMA(5, 2, rα, β)
	@test 𝓒 isa CARMA
	@test 𝓒.p == 5
	@test 𝓒.q == 2
end

function test_sampling_quad()

	rng = MersenneTwister(1234)
	f_min, f_max = 1e-3, 1e2
	for p in 1:5
		for q in 1:p-1
			for i in 1:3
				qa, qb = Pioran.sample_quad(p, q, rng, f_min, f_max)

				ra = quad2roots(qa)
				rb = quad2roots(qb)

				@test Pioran.check_conjugate_pair(ra)
				@test Pioran.check_conjugate_pair(rb)

				@test Pioran.check_roots_bounds(ra, f_min, f_max)
				@test Pioran.check_roots_bounds(rb, f_min, f_max)

				@test Pioran.check_order_imag_roots(ra)
				@test Pioran.check_order_imag_roots(rb)

			end

		end
	end

end

@testset "test_carma" begin
	test_quad2roots32()
	test_roots2coeffs32()
	test_init_CARMA32()
	test_init_CARMA52()
	test_celerite_coefCARMA()
	test_celerite_repr()
	test_CARMA_ACVF()
    test_CARMA_PSD()
	test_sampling_quad()
end
