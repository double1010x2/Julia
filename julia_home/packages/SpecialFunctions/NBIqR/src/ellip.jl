using Base.Math: @horner




@doc raw"""
    ellipk(m)

Computes Complete Elliptic Integral of 1st kind ``K(m)`` for parameter ``m`` given by

```math
\operatorname{ellipk}(m)
= K(m)
= \int_0^{ \frac{\pi}{2} } \frac{1}{\sqrt{1 - m \sin^2 \theta}} \, \mathrm{d}\theta
\quad \text{for} \quad m \in \left( -\infty, 1 \right] \, .
```

External links: [DLMF](https://dlmf.nist.gov/19.2.8), [Wikipedia](https://en.wikipedia.org/wiki/Elliptic_integral#Notational_variants).

See also: [`ellipe(m)`](@ref SpecialFunctions.ellipe).

# Arguments
- `m`: parameter ``m``, restricted to the domain ``(-\infty,1]``, is related to the elliptic modulus ``k`` by ``k^2=m`` and to the modular
    angle ``\alpha`` by ``k=\sin \alpha``.

# Implementation
Using piecewise approximation polynomial as given in
> 'Fast Computation of Complete Elliptic Integrals and Jacobian Elliptic Functions',
> Fukushima, Toshio. (2014). F09-FastEI. Celest Mech Dyn Astr,
> DOI 10.1007/s10569-009-9228-z,
> <https://pdfs.semanticscholar.org/8112/c1f56e833476b61fc54d41e194c962fbe647.pdf>

For ``m<0``, followed by
> Fukushima, Toshio. (2014).
> 'Precise, compact, and fast computation of complete elliptic integrals by piecewise
> minimax rational function approximation'.
> Journal of Computational and Applied Mathematics. 282.
> DOI 10.13140/2.1.1946.6245.,
> <https://www.researchgate.net/publication/267330394>
As suggested in this paper, the domain is restricted to ``(-\infty,1]``.
"""
function ellipk(m::Float64)
    flag_is_m_neg = false
    if m < 0.0
        x               = m / (m-1)         #dealing with negative args
        flag_is_m_neg   = true
    elseif m >= 0.0
        x               = m
        flag_is_m_neg   = false
    end

    if x == 0.0
        return Float64(halfπ)

    elseif x == 1.0
        return Inf

    elseif x > 1.0
        throw(DomainError(m, "`m` must lie between -Inf and 1 ---- Domain: (-Inf,1.0]"))

    elseif 0.0 <= x < 0.1   #Table 2 from paper
        t = x-0.05
        t = @horner(t,
            1.591003453790792180 , 0.416000743991786912 , 0.245791514264103415 ,
            0.179481482914906162 , 0.144556057087555150 , 0.123200993312427711 ,
            0.108938811574293531 , 0.098853409871592910 , 0.091439629201749751 ,
            0.085842591595413900 , 0.081541118718303215)

    elseif 0.1 <= x < 0.2  #Table 3
        t = x-0.15
        t = @horner(t ,
            1.635256732264579992 , 0.471190626148732291 , 0.309728410831499587 ,
            0.252208311773135699 , 0.226725623219684650 , 0.215774446729585976 ,
            0.213108771877348910 , 0.216029124605188282 , 0.223255831633057896 ,
            0.234180501294209925 , 0.248557682972264071 , 0.266363809892617521)

    elseif 0.2 <= x < 0.3 #Table 4
        t = x-0.25
        t = @horner(t ,
            1.685750354812596043 , 0.541731848613280329 , 0.401524438390690257 ,
            0.369642473420889090 , 0.376060715354583645 , 0.405235887085125919 ,
            0.453294381753999079 , 0.520518947651184205 , 0.609426039204995055 ,
            0.724263522282908870 , 0.871013847709812357 , 1.057652872753547036)

    elseif 0.3 <= x < 0.4 #Table 5
        t = x-0.35
        t = @horner(t ,
            1.744350597225613243 , 0.634864275371935304 , 0.539842564164445538 ,
            0.571892705193787391 , 0.670295136265406100 , 0.832586590010977199 ,
            1.073857448247933265 , 1.422091460675497751 , 1.920387183402304829 ,
            2.632552548331654201 , 3.652109747319039160 , 5.115867135558865806 ,
            7.224080007363877411)

    elseif 0.4 <= x < 0.5 #Table 6
        t = x-0.45
        t = @horner(t,
            1.813883936816982644 , 0.763163245700557246 , 0.761928605321595831 ,
            0.951074653668427927 , 1.315180671703161215 , 1.928560693477410941 ,
            2.937509342531378755 , 4.594894405442878062 , 7.330071221881720772 ,
            11.87151259742530180 , 19.45851374822937738 , 32.20638657246426863 ,
            53.73749198700554656 , 90.27388602940998849)

    elseif 0.5 <= x < 0.6 #Table 7
        t = x-0.55
        t = @horner(t ,
            1.898924910271553526 , 0.950521794618244435 , 1.151077589959015808 ,
            1.750239106986300540 , 2.952676812636875180 , 5.285800396121450889 ,
            9.832485716659979747 , 18.78714868327559562 , 36.61468615273698145 ,
            72.45292395127771801 , 145.1079577347069102 , 293.4786396308497026 ,
            598.3851815055010179 , 1228.420013075863451 , 2536.529755382764488)
    elseif 0.6 <= x < 0.7 #Table 8
        t = x-0.65
        t = @horner(t ,
            2.007598398424376302 , 1.248457231212347337 , 1.926234657076479729 ,
            3.751289640087587680 , 8.119944554932045802 , 18.66572130873555361 ,
            44.60392484291437063 , 109.5092054309498377 , 274.2779548232413480 ,
            697.5598008606326163 , 1795.716014500247129 , 4668.381716790389910 ,
            12235.76246813664335 , 32290.17809718320818 , 85713.07608195964685 ,
            228672.1890493117096 , 612757.2711915852774)

    elseif 0.7 <= x < 0.8 #Table 9
        t = x-0.75
        t = @horner(t,
            2.156515647499643235 , 1.791805641849463243 , 3.826751287465713147 ,
            10.38672468363797208 , 31.40331405468070290 , 100.9237039498695416 ,
            337.3268282632272897 , 1158.707930567827917 , 4060.990742193632092 ,
            14454.00184034344795 , 52076.66107599404803 , 189493.6591462156887 ,
            695184.5762413896145 , 2567994.048255284686 , 9541921.966748386322 ,
            35634927.44218076174 , 133669298.4612040871 , 503352186.6866284541 ,
            1901975729.538660119 , 7208915015.330103756)

    elseif 0.8 <= x < 0.85 #Table 10
        t = x-0.825
        t = @horner(t ,
            2.318122621712510589 , 2.616920150291232841 , 7.897935075731355823 ,
            30.50239715446672327 , 131.4869365523528456 , 602.9847637356491617 ,
            2877.024617809972641 , 14110.51991915180325 , 70621.44088156540229 ,
            358977.2665825309926 , 1847238.263723971684 , 9600515.416049214109 ,
            50307677.08502366879 , 265444188.6527127967 , 1408862325.028702687 ,
            7515687935.373774627)

    elseif 0.85 <= x < 0.9 #Table 11
        t = x-0.875
        t = @horner(t,
            2.473596173751343912 , 3.727624244118099310 , 15.60739303554930496 ,
            84.12850842805887747 , 506.9818197040613935 , 3252.277058145123644 ,
            21713.24241957434256 , 149037.0451890932766 , 1043999.331089990839 ,
            7427974.817042038995 , 53503839.67558661151 , 389249886.9948708474 ,
            2855288351.100810619 , 21090077038.76684053 , 156699833947.7902014 ,
            1170222242422.439893 , 8777948323668.937971 , 66101242752484.95041 ,
            499488053713388.7989 , 37859743397240299.20)

    elseif x >= 0.9
        td  = 1-x
        td1 = td-0.05
        qd  = @horner(td,
            0.0, (1.0/16.0), (1.0/32.0), (21.0/1024.0), (31.0/2048.0), (6257.0/524288.0),
            (10293.0/1048576.0), (279025.0/33554432.0), (483127.0/67108864.0),
            (435506703.0/68719476736.0), (776957575.0/137438953472.0) ,
            (22417045555.0/4398046511104.0) , (40784671953.0/8796093022208.0) ,
            (9569130097211.0/2251799813685248.0) , (17652604545791.0/4503599627370496.0))

        kdm = @horner(td1 ,
            1.591003453790792180 , 0.416000743991786912 , 0.245791514264103415 ,
            0.179481482914906162 , 0.144556057087555150 , 0.123200993312427711 ,
            0.108938811574293531 , 0.098853409871592910 , 0.091439629201749751 ,
            0.085842591595413900 , 0.081541118718303215)
        km  = -Base.log(qd) * (kdm * invπ)
        t   = km
    end

    if flag_is_m_neg
        ans = t / sqrt(1.0-m)
        return ans
    else
        return t
    end
end


@doc raw"""
    ellipe(m)

Computes Complete Elliptic Integral of 2nd kind ``E(m)`` for parameter ``m`` given by

```math
\operatorname{ellipe}(m)
= E(m)
= \int_0^{ \frac{\pi}{2} } \sqrt{1 - m \sin^2 \theta} \, \mathrm{d}\theta
\quad \text{for} \quad m \in \left( -\infty, 1 \right] \, .
```

External links: [DLMF](https://dlmf.nist.gov/19.2.8), [Wikipedia](https://en.wikipedia.org/wiki/Elliptic_integral#Complete_elliptic_integral_of_the_second_kind).

See also: [`ellipk(m)`](@ref SpecialFunctions.ellipk).

# Arguments
- `m`: parameter ``m``, restricted to the domain ``(-\infty,1]``, is related to the elliptic modulus ``k`` by ``k^2=m`` and to the modular
    angle ``\alpha`` by ``k=\sin \alpha``.

# Implementation
Using piecewise approximation polynomial as given in
> 'Fast Computation of Complete Elliptic Integrals and Jacobian Elliptic Functions',
> Fukushima, Toshio. (2014). F09-FastEI. Celest Mech Dyn Astr,
> DOI 10.1007/s10569-009-9228-z,
> <https://pdfs.semanticscholar.org/8112/c1f56e833476b61fc54d41e194c962fbe647.pdf>

For ``m<0``, followed by
> Fukushima, Toshio. (2014).
> 'Precise, compact, and fast computation of complete elliptic integrals by piecewise
> minimax rational function approximation'.
> Journal of Computational and Applied Mathematics. 282.
> DOI 10.13140/2.1.1946.6245.,
> <https://www.researchgate.net/publication/267330394>
As suggested in this paper, the domain is restricted to ``(-\infty,1]``.
"""
function ellipe(m::Float64)
    flag_is_m_neg = false
    if m < 0.0
        x               = m / (m-1)         #dealing with negative args
        flag_is_m_neg   = true
    elseif m >= 0.0
        x               = m
        flag_is_m_neg   = false
    end

    if x == 0.0
        return Float64(halfπ)
    elseif x == 1.0
        return 1.0

    elseif x > 1.0
        throw(DomainError(m,"`m` must lie between -Inf and 1 ---- Domain : (-inf,1.0]"))

    elseif 0.0 <= x < 0.1   #Table 2 from paper
        t = x-0.05
        t = @horner(t  ,
            +1.550973351780472328 , -0.400301020103198524 , -0.078498619442941939 ,
            -0.034318853117591992 , -0.019718043317365499 , -0.013059507731993309 ,
            -0.009442372874146547 , -0.007246728512402157 , -0.005807424012956090 ,
            -0.004809187786009338)

    elseif 0.1 <= x < 0.2  #Table 3
        t = x-0.15
        t = @horner(t ,
            +1.510121832092819728 , -0.417116333905867549 , -0.090123820404774569 ,
            -0.043729944019084312 , -0.027965493064761785 , -0.020644781177568105 ,
            -0.016650786739707238 , -0.014261960828842520 , -0.012759847429264803 ,
            -0.011799303775587354 , -0.011197445703074968)

    elseif 0.2 <= x < 0.3 #Table 4
        t = x-0.25
        t = @horner(t ,
            +1.467462209339427155 , -0.436576290946337775 , -0.105155557666942554 ,
            -0.057371843593241730 , -0.041391627727340220 , -0.034527728505280841 ,
            -0.031495443512532783 , -0.030527000890325277 , -0.030916984019238900 ,
            -0.032371395314758122 , -0.034789960386404158)

    elseif 0.3 <= x < 0.4 #Table 5
        t = x-0.35
        t = @horner(t,
            +1.422691133490879171 , -0.459513519621048674 , -0.125250539822061878,
            -0.078138545094409477 , -0.064714278472050002 , -0.062084339131730311,
            -0.065197032815572477 , -0.072793895362578779 , -0.084959075171781003,
            -0.102539850131045997 , -0.127053585157696036 , -0.160791120691274606)

    elseif 0.4 <= x < 0.5 #Table 6
        t = x-0.45
        t = @horner(t  ,
            +1.375401971871116291 , -0.487202183273184837 , -0.153311701348540228 ,
            -0.111849444917027833 , -0.108840952523135768 , -0.122954223120269076 ,
            -0.152217163962035047 , -0.200495323642697339 , -0.276174333067751758 ,
            -0.393513114304375851 , -0.575754406027879147 , -0.860523235727239756 ,
            -1.308833205758540162)

    elseif 0.5 <= x < 0.6 #Table 7
        t = x-0.55
        t = @horner(t  ,
            +1.325024497958230082 , -0.521727647557566767 , -0.194906430482126213 ,
            -0.171623726822011264 , -0.202754652926419141 , -0.278798953118534762 ,
            -0.420698457281005762 , -0.675948400853106021 , -1.136343121839229244 ,
            -1.976721143954398261 , -3.531696773095722506 , -6.446753640156048150 ,
            -11.97703130208884026)
    elseif 0.6 <= x < 0.7 #Table 8
        t = x-0.65
        t = @horner(t,
            +1.270707479650149744 , -0.566839168287866583 , -0.262160793432492598 ,
            -0.292244173533077419 , -0.440397840850423189 , -0.774947641381397458 ,
            -1.498870837987561088 , -3.089708310445186667 , -6.667595903381001064 ,
            -14.89436036517319078 , -34.18120574251449024 , -80.15895841905397306 ,
            -191.3489480762984920 , -463.5938853480342030 , -1137.380822169360061)

    elseif 0.7 <= x < 0.8 #Table 9
        t = x-0.75
        t = @horner(t,
            +1.211056027568459525 , -0.630306413287455807 , -0.387166409520669145 ,
            -0.592278235311934603 , -1.237555584513049844 , -3.032056661745247199 ,
            -8.181688221573590762 , -23.55507217389693250 , -71.04099935893064956 ,
            -221.8796853192349888 , -712.1364793277635425 , -2336.125331440396407 ,
            -7801.945954775964673 , -26448.19586059191933 , -90799.48341621365251 ,
            -315126.0406449163424 , -1104011.344311591159)

    elseif 0.8 <= x < 0.85 #Table 10
        t = x-0.825
        t = @horner(t,
            +1.161307152196282836 , -0.701100284555289548 , -0.580551474465437362 ,
            -1.243693061077786614 , -3.679383613496634879 , -12.81590924337895775 ,
            -49.25672530759985272 , -202.1818735434090269 , -869.8602699308701437 ,
            -3877.005847313289571 , -17761.70710170939814 , -83182.69029154232061 ,
            -396650.4505013548170 , -1920033.413682634405)

    elseif 0.85 <= x < 0.9 #Table 11
        t = x-0.875
        t =  @horner(t,
            +1.124617325119752213 , -0.770845056360909542 , -0.844794053644911362 ,
            -2.490097309450394453 , -10.23971741154384360 , -49.74900546551479866 ,
            -267.0986675195705196 , -1532.665883825229947 , -9222.313478526091951 ,
            -57502.51612140314030 , -368596.1167416106063 , -2415611.088701091428 ,
            -16120097.81581656797 , -109209938.5203089915 , -749380758.1942496220 ,
            -5198725846.725541393 , -36409256888.12139973)

    elseif x >= 0.9
        td  = 1-x
        td1 = td-0.05

        kdm = @horner(td1 ,
            1.591003453790792180 , 0.416000743991786912 , 0.245791514264103415 ,
            0.179481482914906162 , 0.144556057087555150 , 0.123200993312427711 ,
            0.108938811574293531 , 0.098853409871592910 , 0.091439629201749751 ,
            0.085842591595413900 , 0.081541118718303215)
        edm = @horner(td1  ,
            +1.550973351780472328 , -0.400301020103198524 , -0.078498619442941939 ,
            -0.034318853117591992 , -0.019718043317365499 , -0.013059507731993309 ,
            -0.009442372874146547 , -0.007246728512402157 , -0.005807424012956090 ,
            -0.004809187786009338)
        hdm = kdm - edm
        km  = ellipk(Float64(x))
        #em =  km + (pi/2 - km*edm)/kdm
        em  = (halfπ + km*hdm) / kdm   #to avoid precision loss near 1
        t   = em
    end
    if flag_is_m_neg
        return t * sqrt(1.0-m)
    else
        return t
    end
end

for f in (:ellipk,:ellipe)
    @eval begin
        ($f)(x::Float16)        = Float16(($f)(Float64(x)))
	    ($f)(x::Float32)        = Float32(($f)(Float64(x)))
        ($f)(x::Real)           = ($f)(float(x))
	    ($f)(x::AbstractFloat)  = throw(MethodError($f, (x, "")))
    end
end
