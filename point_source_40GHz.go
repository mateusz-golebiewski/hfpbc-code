//SPIN WAVE PROPAGATION FROM A POINT SOURCE

// number of unit cells in each direction
Nx := 3000
Ny := 328
Nz := 1

// unit cells size [m]
cx := 5e-09
cy := 5e-09
cz := 5e-09

B0 := 1.1 // static external magnetic field [T]

setgridsize(Nx, Ny, Nz)
setcellsize(cx, cy, cz)

g := yrange(-Ny*cy/2, Ny*cy/2)

setgeom(g)
Snapshot(geom)
DefRegion((0), g) 
EdgeSmooth = 8
SetPBC(0, 0, 0) // periodic boundary condition
DefRegion(0, g)

m = uniform(0, 0, 1) // initial magnetization state
B_ext = vector(0, 0, B0) // ext. magn. field direction

Ms_YIG 	:= 0.137e+06 // saturation magnetization characteristic for yttrium iron garnet (YIG)
A_YIG 	:= 0.4e-11 // exchange stiffness characteristic for YIG

Aex 	= A_YIG
Msat 	= Ms_YIG

grad_0  	:= xrange(1,0)
grad_left  	:= xrange(1,0)
grad_right  := xrange(1,0)
grad_up  	:= yrange(1,0)
grad_down  	:= yrange(1,0)

// Absorbing boundary condition
grad := 100
gr_step := 4
for i:=2; i<(grad+2); i++{
	grad_up = yrange( ( (Ny*cy/2)-gr_step*(grad)*cx+i*gr_step*cx), ( (Ny*cy/2)-gr_step*(grad)*cy+gr_step*(i+1)*cy) )
	grad_down  = yrange( (-(Ny*cy/2)+gr_step*(grad)*cy-gr_step*(i+1)*cy), (-(Ny*cy/2)+gr_step*(grad)*cy-gr_step*i*cy) )
	grad_left = xrange( (-(Nx*cx/2)+gr_step*(grad)*cx-gr_step*(i+1)*cx), (-(Nx*cx/2)+gr_step*(grad)*cx-gr_step*i*cx) )
	grad_right = xrange( ((Nx*cx/2)-gr_step*(grad)*cx+gr_step*i*cx), ((Nx*cx/2)-gr_step*(grad)*cx+gr_step*(i+1)*cx) )
  	grad_0  = grad_left.add(grad_right)
  	grad_up.add(grad_down.add(grad_right.add(grad_left)))
	DefRegion(i, grad_0 ) 
}

Snapshot(regions)
saveas(m, "m_init")
relax()
saveas(m, "m_stab")
Snapshot(m)
saveas(B_eff, "B_eff")
Snapshot(B_eff)
saveas(B_demag, "B_demag")
Snapshot(B_demag)

print("static simulation")

maskPointLike := newSlice(3, Nx, Ny, 1)
x00 := Nx/2
y00 := Ny/2
sgm2 := (2*0.2)*50.0
rho := 0.0
Amp1 := 0.0 
k0 := 2*pi/(10*cx)
print(rho, sgm2)
for x:=0; x<Nx; x++{
	for y:=0; y<Ny; y++{
		rho = sqrt( pow( (x-x00), 2) + pow( (y-y00), 2) )
		if (rho<=2) {
			Amp1 = 1.0
		} else {
			Amp1 = 0.0
		}
		maskPointLike.set(0, x, y, 0, 0) // component(s) perpendicular to the static field
		maskPointLike.set(1, x, y, 0, Amp1)
		maskPointLike.set(2, x, y, 0, 0)
	}
}

relax() // relaxation of the system

alpha0 := 0.0001 // damping constant
alpha = alpha0
for i:=2; i<(grad+2); i++{
	dmp := alpha0 + 0.5*(i/grad)*(i/grad)
    alpha.setRegion(i, dmp)
}

alpha.setRegion(0, alpha0)
snapshot(alpha)
relax()

f0 := 40e09 // SW frequency [Hz]

MaxDt = 1/f0/200  // 8.333e-12 [s]
MinDt = 1/f0/2000
MaxErr = 0.05e-06

t_sampl := 1/f0/4

Snapshot(m)
t = 0
B_ext.add( maskPointLike, 0.03*B0*sin(2*pi*f0*t) )	

t_steadyState := 15e-9 // time until the system reaches a steady state 
					   // (fully developed interference pattern)
					   
autosnapshot(m.comp(0), t_steadyState/100)
run(t_steadyState)

autosave(m.comp(0), t_sampl)
run(110*t_sampl)

saveas(m, "m_final")
