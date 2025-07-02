in vec3 aColor;
in vec4 aPosition;
out vec4 C;
uniform sampler2D iChannel0;
uniform sampler2D iChannel1;  //Chao
uniform sampler2D iChannel2;
uniform sampler2D iChannel3; // Quadro
uniform sampler2D iChannel4; // Bandeira USP
uniform sampler2D iChannel5; // Bandeira Brasil
uniform sampler2D iChannel6; // Bandeira SP
uniform vec2 iResolution;
uniform vec4 iMouse;
uniform float iTime;
uniform int iFrame;

#define MAX_STEPS 100
#define MAX_DIST 200.
#define SURF_DIST .01
#define EPSILON .01
#define PI 3.14159265359
float dot2( in vec2 v ) { return dot(v,v); }
float dot2( in vec3 v ) { return dot(v,v); }
float ndot( in vec2 a, in vec2 b ) { return a.x*b.x - a.y*b.y; }

// vec3 Sky( vec3 ray )
// {
//         return mix( vec3(.8), vec3(0), exp2(-(1.0/max(ray.y,.01))*vec3(.4,.6,1.0)) );
// }

vec3 Sky( vec3 ray )
{
    // Gradiente: mais claro para cima, mais escuro para baixo
    float t = clamp(ray.y * 0.5 + 0.5, 0.0, 1.0);
    // Branco amarelado no teto, cinza claro no chão
    return mix(vec3(0.7, 0.7, 0.75), vec3(1.0, 0.98, 0.92), t);
}

// noise
float noise(vec2 pos)
{
        return fract( sin( dot(pos*0.001 ,vec2(24.12357, 36.789) ) ) * 12345.123);
}


// blur noise
float smooth_noise(vec2 pos)
{
        return   ( noise(pos + vec2(1,1)) + noise(pos + vec2(1,1)) + noise(pos + vec2(1,1)) + noise(pos + vec2(1,1)) ) / 16.0
                   + ( noise(pos + vec2(1,0)) + noise(pos + vec2(-1,0)) + noise(pos + vec2(0,1)) + noise(pos + vec2(0,-1)) ) / 8.0
           + noise(pos) / 4.0;
}


// linear interpolation
float interpolate_noise(vec2 pos)
{
        float	a, b, c, d;

        a = smooth_noise(floor(pos));
        b = smooth_noise(vec2(floor(pos.x+1.0), floor(pos.y)));
        c = smooth_noise(vec2(floor(pos.x), floor(pos.y+1.0)));
        d = smooth_noise(vec2(floor(pos.x+1.0), floor(pos.y+1.0)));

        a = mix(a, b, fract(pos.x));
        b = mix(c, d, fract(pos.x));
        a = mix(a, b, fract(pos.y));

        return a;
}



float perlin_noise(vec2 pos)
{
        float	n;

        n = interpolate_noise(pos*0.0625)*0.5;
        n += interpolate_noise(pos*0.125)*0.25;
        n += interpolate_noise(pos*0.025)*0.225;
        n += interpolate_noise(pos*0.05)*0.0625;
        n += interpolate_noise(pos)*0.03125;
        return n;
}



const vec3 COLOR_BACKGROUND = vec3(0.25, 0.1, 0.15);

struct Surface
{
    float sd;
    vec3 color;
    float Ka;
    float Kd;
    float Ks;
    int id;
};

mat2 rotate2d(float theta)
{
    float co = cos(theta);
    float s=sin(theta);
    return mat2(co,-s,s,co);
}

mat4 trans (vec3 t)
{
    mat4 mat = mat4 (vec4 (1., .0, .0, .0),
                     vec4 (.0, 1., .0, .0),
                     vec4 (.0, .0, 1., .0),
                     vec4 (t.x, t.y, t.z, 1.));
    return mat;
}

vec3 opTransf (vec3 p, mat4 m)
{
    return vec4 (m * vec4 (p, 1.)).xyz;
}

mat4 rotX (in float angle)
{
    float rad = radians (angle);
    float c = cos (rad);
    float s = sin (rad);

    mat4 mat = mat4 (vec4 (1.0, 0.0, 0.0, 0.0),
                     vec4 (0.0,   c,   s, 0.0),
                     vec4 (0.0,  -s,   c, 0.0),
                     vec4 (0.0, 0.0, 0.0, 1.0));

    return mat;
}

mat4 rotY (in float angle)
{
    float rad = radians (angle);
    float c = cos (rad);
    float s = sin (rad);

    mat4 mat = mat4 (vec4 (  c, 0.0,  -s, 0.0),
                     vec4 (0.0, 1.0, 0.0, 0.0),
                     vec4 (  s, 0.0,   c, 0.0),
                     vec4 (0.0, 0.0, 0.0, 1.0));

    return mat;
}

mat4 rotZ (in float angle)
{
    float rad = radians (angle);
    float c = cos (rad);
    float s = sin (rad);

    mat4 mat = mat4 (vec4 (  c,   s, 0.0, 0.0),
                     vec4 ( -s,   c, 0.0, 0.0),
                     vec4 (0.0, 0.0, 1.0, 0.0),
                     vec4 (0.0, 0.0, 0.0, 1.0));

    return mat;
}


float sphereDist(vec3 p, float r)
{
    vec3 center = p;
    return (length(center)-r*r);
}
float heartDist(vec3 p, vec3 c, float r)
{

    vec3 center = p-c;

    center.x=abs(center.x);
        center.y*=0.9;
        center.z*=(1.1-center.y/15.0);
        center.y+=-0.3-center.x*sqrt(((2.0-center.x)/2.0));
        r+=0.12*pow(0.5+0.5*sin(2.0*PI*iTime+center.y/0.75),4.0);
    return (length(center)-r*r);
}

float planeDist(vec3 p, vec3 Normal,float D)
{
    return p.x*Normal.x + p.y*Normal.y + p.z*Normal.z -D;// - .1 * sin (4. * p.x) * cos (4. * p.z);
}


float boxDist( vec3 p, vec3 b )
{
    vec3 d = abs(p) - b;
    return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}
float boxRoundDist(vec3 p, vec3 s, float r) {
    vec3 d = abs(p) - (s - r);
    return length(max(d, 0.0)) - r
        + min(max(d.x, max(d.y, d.z)), 0.0);
}


float boxFrameDist( vec3 p, vec3 b, float e )
{
       p = abs(p  )-b;
  vec3 q = abs(p+e)-e;

  return min(min(
      length(max(vec3(p.x,q.y,q.z),0.0))+min(max(p.x,max(q.y,q.z)),0.0),
      length(max(vec3(q.x,p.y,q.z),0.0))+min(max(q.x,max(p.y,q.z)),0.0)),
      length(max(vec3(q.x,q.y,p.z),0.0))+min(max(q.x,max(q.y,p.z)),0.0));
}
float ellipsoidDist( in vec3 p, in vec3 r ) // approximated
{
    float k0 = length(p/r);
    float k1 = length(p/(r*r));
    return k0*(k0-1.0)/k1;
}

float torusDist( vec3 p, vec2 t )
{
    return length( vec2(length(p.xz)-t.x,p.y) )-t.y;
}

float cappedTorusDist(in vec3 p, in vec2 sc, in float ra, in float rb)
{
    p.x = abs(p.x);
    float k = (sc.y*p.x>sc.x*p.y) ? dot(p.xy,sc) : length(p.xy);
    return sqrt( dot(p,p) + ra*ra - 2.0*ra*k ) - rb;
}

float hexPrismDist( vec3 p, vec2 h )
{
    vec3 q = abs(p);

    const vec3 k = vec3(-0.8660254, 0.5, 0.57735);
    p = abs(p);
    p.xy -= 2.0*min(dot(k.xy, p.xy), 0.0)*k.xy;
    vec2 d = vec2(
       length(p.xy - vec2(clamp(p.x, -k.z*h.x, k.z*h.x), h.x))*sign(p.y - h.x),
       p.z-h.y );
    return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float octogonPrismDist( in vec3 p, in float r, float h )
{
  const vec3 k = vec3(-0.9238795325,   // sqrt(2+sqrt(2))/2
                       0.3826834323,   // sqrt(2-sqrt(2))/2
                       0.4142135623 ); // sqrt(2)-1
  // reflections
  p = abs(p);
  p.xy -= 2.0*min(dot(vec2( k.x,k.y),p.xy),0.0)*vec2( k.x,k.y);
  p.xy -= 2.0*min(dot(vec2(-k.x,k.y),p.xy),0.0)*vec2(-k.x,k.y);
  // polygon side
  p.xy -= vec2(clamp(p.x, -k.z*r, k.z*r), r);
  vec2 d = vec2( length(p.xy)*sign(p.y), p.z-h );
  return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float capsuleDist( vec3 p, vec3 a, vec3 b, float r )
{
        vec3 pa = p-a, ba = b-a;
        float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
        return length( pa - ba*h ) - r;
}

float roundConeDist( in vec3 p, in float r1, float r2, float h )
{
    vec2 q = vec2( length(p.xz), p.y );

    float b = (r1-r2)/h;
    float a = sqrt(1.0-b*b);
    float k = dot(q,vec2(-b,a));

    if( k < 0.0 ) return length(q) - r1;
    if( k > a*h ) return length(q-vec2(0.0,h)) - r2;

    return dot(q, vec2(a,b) ) - r1;
}

float roundConeDist2(vec3 p, vec3 a, vec3 b, float r1, float r2)
{
    // sampling independent computations (only depend on shape)
    vec3  ba = b - a;
    float l2 = dot(ba,ba);
    float rr = r1 - r2;
    float a2 = l2 - rr*rr;
    float il2 = 1.0/l2;

    // sampling dependant computations
    vec3 pa = p - a;
    float y = dot(pa,ba);
    float z = y - l2;
    float x2 = dot2( pa*l2 - ba*y );
    float y2 = y*y*l2;
    float z2 = z*z*l2;

    // single square root!
    float k = sign(rr)*rr*rr*x2;
    if( sign(z)*a2*z2 > k ) return  sqrt(x2 + z2)        *il2 - r2;
    if( sign(y)*a2*y2 < k ) return  sqrt(x2 + y2)        *il2 - r1;
                            return (sqrt(x2*a2*il2)+y*rr)*il2 - r1;
}

float triPrismDist( vec3 p, vec2 h )
{
    const float k = sqrt(3.0);
    h.x *= 0.5*k;
    p.xy /= h.x;
    p.x = abs(p.x) - 1.0;
    p.y = p.y + 1.0/k;
    if( p.x+k*p.y>0.0 ) p.xy=vec2(p.x-k*p.y,-k*p.x-p.y)/2.0;
    p.x -= clamp( p.x, -2.0, 0.0 );
    float d1 = length(p.xy)*sign(-p.y)*h.x;
    float d2 = abs(p.z)-h.y;
    return length(max(vec2(d1,d2),0.0)) + min(max(d1,d2), 0.);
}

// vertical
float cylinderVerticalDist( vec3 p, vec2 h )
{
    vec2 d = abs(vec2(length(p.xz),p.y)) - h;
    return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

// arbitrary orientation
float cylinderDist(vec3 p, vec3 a, vec3 b, float r)
{
    vec3 pa = p - a;
    vec3 ba = b - a;
    float baba = dot(ba,ba);
    float paba = dot(pa,ba);

    float x = length(pa*baba-ba*paba) - r*baba;
    float y = abs(paba-baba*0.5)-baba*0.5;
    float x2 = x*x;
    float y2 = y*y*baba;
    float d = (max(x,y)<0.0)?-min(x2,y2):(((x>0.0)?x2:0.0)+((y>0.0)?y2:0.0));
    return sign(d)*sqrt(abs(d))/baba;
}

// vertical
float coneVerticalDist( in vec3 p, in vec2 c, float h )
{
    vec2 q = h*vec2(c.x,-c.y)/c.y;
    vec2 w = vec2( length(p.xz), p.y );

        vec2 a = w - q*clamp( dot(w,q)/dot(q,q), 0.0, 1.0 );
    vec2 b = w - q*vec2( clamp( w.x/q.x, 0.0, 1.0 ), 1.0 );
    float k = sign( q.y );
    float d = min(dot( a, a ),dot(b, b));
    float s = max( k*(w.x*q.y-w.y*q.x),k*(w.y-q.y)  );
        return sqrt(d)*sign(s);
}

float cappedConeDist( in vec3 p, in float h, in float r1, in float r2 )
{
    vec2 q = vec2( length(p.xz), p.y );

    vec2 k1 = vec2(r2,h);
    vec2 k2 = vec2(r2-r1,2.0*h);
    vec2 ca = vec2(q.x-min(q.x,(q.y < 0.0)?r1:r2), abs(q.y)-h);
    vec2 cb = q - k1 + k2*clamp( dot(k1-q,k2)/dot2(k2), 0.0, 1.0 );
    float s = (cb.x < 0.0 && ca.y < 0.0) ? -1.0 : 1.0;
    return s*sqrt( min(dot2(ca),dot2(cb)) );
}

float cappedConeDist_a(vec3 p, vec3 a, vec3 b, float ra, float rb)
{
    float rba  = rb-ra;
    float baba = dot(b-a,b-a);
    float papa = dot(p-a,p-a);
    float paba = dot(p-a,b-a)/baba;

    float x = sqrt( papa - paba*paba*baba );

    float cax = max(0.0,x-((paba<0.5)?ra:rb));
    float cay = abs(paba-0.5)-0.5;

    float k = rba*rba + baba;
    float f = clamp( (rba*(x-ra)+paba*baba)/k, 0.0, 1.0 );

    float cbx = x-ra - f*rba;
    float cby = paba - f;

    float s = (cbx < 0.0 && cay < 0.0) ? -1.0 : 1.0;

    return s*sqrt( min(cax*cax + cay*cay*baba,
                       cbx*cbx + cby*cby*baba) );
}

// c is the sin/cos of the desired cone angle
float solidAngleDist(vec3 pos, vec2 c, float ra)
{
    vec2 p = vec2( length(pos.xz), pos.y );
    float l = length(p) - ra;
        float m = length(p - c*clamp(dot(p,c),0.0,ra) );
    return max(l,m*sign(c.y*p.x-c.x*p.y));
}

float octahedronDist(vec3 p, float s)
{
    p = abs(p);
    float m = p.x + p.y + p.z - s;

    // exact distance
    #if 0
    vec3 o = min(3.0*p - m, 0.0);
    o = max(6.0*p - m*2.0 - o*3.0 + (o.x+o.y+o.z), 0.0);
    return length(p - s*o/(o.x+o.y+o.z));
    #endif

    // exact distance
    #if 1
        vec3 q;
         if( 3.0*p.x < m ) q = p.xyz;
    else if( 3.0*p.y < m ) q = p.yzx;
    else if( 3.0*p.z < m ) q = p.zxy;
    else return m*0.57735027;
    float k = clamp(0.5*(q.z-q.y+s),0.0,s);
    return length(vec3(q.x,q.y-s+k,q.z-k));
    #endif

    // bound, not exact
    #if 0
        return m*0.57735027;
    #endif
}

float pyramidDist( in vec3 p, in float h )
{
    float m2 = h*h + 0.25;

    // symmetry
    p.xz = abs(p.xz);
    p.xz = (p.z>p.x) ? p.zx : p.xz;
    p.xz -= 0.5;

    // project into face plane (2D)
    vec3 q = vec3( p.z, h*p.y - 0.5*p.x, h*p.x + 0.5*p.y);

    float s = max(-q.x,0.0);
    float t = clamp( (q.y-0.5*p.z)/(m2+0.25), 0.0, 1.0 );

    float a = m2*(q.x+s)*(q.x+s) + q.y*q.y;
        float b = m2*(q.x+0.5*t)*(q.x+0.5*t) + (q.y-m2*t)*(q.y-m2*t);

    float d2 = min(q.y,-q.x*m2-q.y*0.5) > 0.0 ? 0.0 : min(a,b);

    // recover 3D and scale, and add sign
    return sqrt( (d2+q.z*q.z)/m2 ) * sign(max(q.z,-p.y));;
}

// la,lb=semi axis, h=height, ra=corner
float rhombusDist(vec3 p, float la, float lb, float h, float ra)
{
    p = abs(p);
    vec2 b = vec2(la,lb);
    float f = clamp( (ndot(b,b-2.0*p.xz))/dot(b,b), -1.0, 1.0 );
        vec2 q = vec2(length(p.xz-0.5*b*vec2(1.0-f,1.0+f))*sign(p.x*b.y+p.z*b.x-b.x*b.y)-ra, p.y-h);
    return min(max(q.x,q.y),0.0) + length(max(q,0.0));
}

float horseshoeDist( in vec3 p, in vec2 c, in float r, in float le, vec2 w )
{
    p.x = abs(p.x);
    float l = length(p.xy);
    p.xy = mat2(-c.x, c.y,
              c.y, c.x)*p.xy;
    p.xy = vec2((p.y>0.0 || p.x>0.0)?p.x:l*sign(-c.x),
                (p.x>0.0)?p.y:l );
    p.xy = vec2(p.x,abs(p.y-r))-vec2(le,0.0);

    vec2 q = vec2(length(max(p.xy,0.0)) + min(0.0,max(p.x,p.y)),p.z);
    vec2 d = abs(q) - w;
    return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

Surface unionS(Surface s1,Surface s2)
{
    if(s1.sd<=s2.sd)
        return s1;
    else
        return s2;
}

Surface intersectionS(Surface s1,Surface s2)
{
    if(s1.sd<=s2.sd)
        return s2;
    else
        return s1;
}

Surface subtractionS(Surface s1,Surface s2)
{
    Surface Ret;
    if(s1.sd<=(-s2.sd))
    {
        Ret.color=s2.color;
        Ret.Ka=s2.Ka;
        Ret.Kd=s2.Kd;
        Ret.Ks=s2.Ks;
        Ret.sd=-s2.sd;
        Ret.id = s2.id;  // manter id do corte

        return Ret;
    }
    else
        return s1;
}

// Surface csgObject(vec3 p)
// {
//     float t = iTime;

//     // Aplica rotação e translação no queijo
//     mat4 m = trans(vec3(0.0, -0.40, 0.0)) * rotY(40.0 * t);
//     p = opTransf(p, m);

//     // -----------------------------------
//     // QUEIJO: cilindro principal
//     Surface queijo;
//     queijo.sd = cylinderVerticalDist(p, vec2(2.0, 0.3)); // raio=2, altura=0.3
//     queijo.color = vec3(0.0, 0.0, 1.0);
//     queijo.Ka = 0.3; queijo.Kd = 0.5; queijo.Ks = 0.2;

//     // -----------------------------------
//     // FATIA radial (como pizza)

//     // Ângulo da abertura da fatia
//     float angle = radians(120.0); // fatia de ±10º (total de 20)

//     // Primeiro plano lateral (ângulo positivo)
//     vec3 n1 = normalize(vec3(sin(angle / 2.0), 0.0, -cos(angle / 2.0)));
//     float d1 = 0.0;
//     float corte1 = planeDist(p, n1, d1);

//     // Segundo plano lateral (ângulo negativo)
//     vec3 n2 = normalize(vec3(-sin(angle / 2.0), 0.0, -cos(angle / 2.0)));
//     float d2 = 0.0;
//     float corte2 = planeDist(p, n2, d2);

//     // Corte interno: limita a fatia até o centro
//     float corte3 = -cylinderVerticalDist(p, vec2(2.0, 0.3)); // pega só dentro do queijo

//     // Interseção dos planos define o volume da fatia
//     float fatia = max(max(corte1, corte2), corte3);

//     Surface corte;
//     corte.sd = fatia;
//     corte.color = vec3(0.0, 0.0, 0.0); 
//     corte.Ka = 0.0; corte.Kd = 0.0; corte.Ks = 0.0;

//     // Subtrai a fatia do queijo
//     Surface resultado = subtractionS(queijo, corte);

//     return resultado;
// }

// Surface csgObject(vec3 p)
// {
//     float t = iTime;

//     // Aplica rotação e translação
//     mat4 m = trans(vec3(0.0, -0.40, 0.0)) * rotY(40.0 * t);
//     p = opTransf(p, m);

//     // Define o cilindro de queijo
//     Surface queijo;
//     queijo.sd = cylinderVerticalDist(p, vec2(2.0, 0.3)); // raio, altura
//     queijo.color = vec3(1.2, 1.0, 0.3);
//     queijo.Ka = 0.3; queijo.Kd = 0.5; queijo.Ks = 0.2;

//     // Define o prisma triangular a ser removido (a fatia)
//     vec3 fatiaPos = p;

//     // Gira o espaço da fatia para apontar para o centro (em torno de Y)
//     float angulo = radians(120.0); // ângulo da fatia, ajuste para abrir mais ou menos
//     mat2 rot = mat2(cos(angulo), -sin(angulo), sin(angulo), cos(angulo));
//     fatiaPos.xz = rot * fatiaPos.xz;

//     // Desloca a fatia radialmente
//     fatiaPos -= vec3(2.0, 0.0, 0.0); // deslocamento radial

//     float fatia = triPrismDist(fatiaPos, vec2(4.0, 0.3)); // base grande, altura da cena

//     Surface corte;
//     corte.sd = fatia;
//     corte.color = vec3(0.0);
//     corte.Ka = 0.5; corte.Kd = 0.5; corte.Ks = 0.5;

//     // Subtrai a fatia do cilindro
//     Surface resultado = subtractionS(queijo, corte);

//     return resultado;
// }

// Surface csgObject(vec3 p)
// {
//     float t = iTime;

//     // Aplica rotação e translação no queijo
//     mat4 m = trans(vec3(0.0, -0.40, 0.0)) * rotY(40.0 * t);
//     p = opTransf(p, m);

//     // Cilindro principal do queijo
//     Surface queijo;
//     queijo.sd = cylinderVerticalDist(p, vec2(2.0, 0.3)); // raio=2, altura=0.3
//     queijo.color = vec3(1.2, 1.0, 0.3);
//     queijo.Ka = 0.3; queijo.Kd = 0.5; queijo.Ks = 0.2;

//     // Ângulo da fatia (setor)
//     float angle = radians(120.0);

//     // Planos laterais da fatia (definem setor angular)
//     vec3 n1 = normalize(vec3(sin(angle / 2.0), 0.0, -cos(angle / 2.0)));
//     vec3 n2 = normalize(vec3(-sin(angle / 2.0), 0.0, -cos(angle / 2.0)));
//     float corte1 = planeDist(p, n1, 0.0);
//     float corte2 = planeDist(p, n2, 0.0);

//     // Limita a fatia até o centro do cilindro (negando cilindro)
//     float corte3 = -cylinderVerticalDist(p, vec2(2.0, 0.3));

//     // Interseção dos planos forma a fatia
//     float fatia = max(max(corte1, corte2), corte3);


//     Surface corte;
//     corte.sd = fatia;
//     // Sem cor nem iluminação para corte (espaço vazio)
//     corte.color = vec3(0.0);
    
//     corte.Ka = 0.0; corte.Kd = 0.0; corte.Ks = 0.0;
//     // Subtração da fatia do queijo
//     Surface resultado = subtractionS(queijo, corte);

//     return resultado;
// }

//CODIGO QUEIJO FATIA CLARA
Surface csgObject(vec3 p)
{
    float t = iTime;

    // Armazena ponto original (mundo) antes de transformar
    vec3 pMundo = p;

    // Rotação e translação do queijo
    // Deslocamento de 5 pixels para a esquerda
    vec3 deslocamentoQueijo = vec3(5.0, -0.40, 0.0);
    mat4 m = rotY(40.0 * t) * trans(deslocamentoQueijo);
    p = opTransf(p, m); // esse "p" será usado só para o queijo

    // =========================
    // QUEIJO COM CORTES (igual estava)
    // =========================
    Surface queijo;
    queijo.sd = cylinderVerticalDist(p, vec2(1.7, 0.3));
    queijo.color = vec3(1.2, 1.0, 0.3);
    queijo.Ka = 0.3; queijo.Kd = 0.5; queijo.Ks = 0.2;

    float angle = radians(40.0);
    vec3 n1 = normalize(vec3(sin(angle), 0.0, -cos(angle)));
    vec3 n2 = normalize(vec3(-sin(angle), 0.0, -cos(angle)));
    float corte1 = planeDist(p, n1, 0.0);
    float corte2 = planeDist(p, n2, 0.0);
    float corte3 = -cylinderVerticalDist(p, vec2(2.0, 0.3));
    float fatia = max(max(corte1, corte2), corte3);

    Surface corte;
    corte.sd = fatia;
    corte.color = vec3(0.0);
    corte.Ka = 0.5; corte.Kd = 0.7; corte.Ks = 0.0;
    corte.id = 6;

    Surface resultado = subtractionS(queijo, corte);

    // --- ÁRVORE NO CENTRO DO QUEIJO (alta e com mais ramos) ---
    vec3 centroArvore = vec3(0.0, 0.3, 0.0); // centro do queijo, em cima dele

    // Tronco (cilindro vertical, mais alto)
    vec3 pTronco = p - centroArvore;
    float troncoAltura = 1.1;
    float troncoSD = cylinderVerticalDist(pTronco, vec2(0.08, troncoAltura));

    // Copa principal (esfera maior em cima do tronco)
    vec3 copaPos = centroArvore + vec3(0.0, troncoAltura, 0.0);
    float copaSD = length(p - copaPos) - 0.32;

    // Coordenadas UV esféricas para textura
    vec3 dir = normalize(p - copaPos);
    float u = 0.5 + atan(dir.z, dir.x) / (2.0 * PI);
    float v = 0.5 - asin(dir.y) / PI;
    vec2 uvCopa = vec2(u, v);

    // Junta tronco e copa
    float arvoreSD = min(troncoSD, copaSD);

    Surface Arvore;
    Arvore.sd = arvoreSD;
    // Corrigido: aplica textura na copa usando máscara
    float copaMask = step(copaSD, troncoSD); // 1.0 se copaSD <= troncoSD
    Arvore.color = mix(vec3(0.4, 0.2, 0.05), texture(iChannel2, uvCopa).rgb, copaMask);
    Arvore.Ka = 0.3; Arvore.Kd = 0.6; Arvore.Ks = 0.2;
    Arvore.id = 60;

    // Adiciona a árvore ao queijo
    resultado = unionS(resultado, Arvore);
    return resultado;
}

// //CODIGO QUEIJO FATIA RETANGULAR
// Surface csgObject(vec3 p)
// {
//     float t = iTime;

//     // Aplica rotação e translação no queijo
//     mat4 m = trans(vec3(0.0, -0.40, 0.0)) * rotY(40.0 * t);
//     p = opTransf(p, m);

//     // Cilindro principal do queijo
//     Surface queijo;
//     queijo.sd = cylinderVerticalDist(p, vec2(2.0, 0.3)); // raio=2, altura=0.3
//     queijo.color = vec3(1.2, 1.0, 0.3);
//     queijo.Ka = 0.3; queijo.Kd = 0.5; queijo.Ks = 0.2;

//     // --- FATIA TRIANGULAR ---
//     // Gira o espaço da fatia para alinhar com o centro do queijo
//     float angFatia = radians(80.0); // ângulo da fatia (ajuste para mais/menos abertura)
//     mat2 rot = mat2(cos(angFatia), -sin(angFatia), sin(angFatia), cos(angFatia));
//     vec3 pFatia = p;
//     pFatia.xz = rot * pFatia.xz;
//     // Desloca a fatia para a borda do queijo
//     pFatia -= vec3(2.0, 0.0, 0.0);

//     // Prisma triangular: base grande, altura igual ao queijo
//     float fatia = triPrismDist(pFatia, vec2(4.0, 0.3)); // base, altura

//     Surface corte;
//     corte.sd = fatia;
//     corte.color = vec3(1.0, 0.9, 0.5); // cor da parte interna (opcional)
//     corte.Ka = 0.3; corte.Kd = 0.5; corte.Ks = 0.2;
//     corte.id = 6;

//     // Subtrai a fatia do queijo
//     Surface resultado = subtractionS(queijo, corte);

//     return resultado;
// }
// //TENTATIVA QUEIJO
// Surface csgObject(vec3 p)
// {
//     float t = iTime;

//     // Gira em torno do eixo Y e sobe um pouco
//     mat4 m = rotY(40.0 * t) * trans(vec3(0.0, -0.40, 0.0));
//     p = opTransf(p, m);

//     // Cilindro de queijo
//     Surface queijo;
//     queijo.sd = cylinderVerticalDist(p, vec2(2.0, 0.3)); // raio, altura
//     queijo.color = vec3(1.2, 1.0, 0.3); // amarelo forte
//     queijo.Ka = 0.3; queijo.Kd = 0.5; queijo.Ks = 0.2;

//     return queijo;
// }



Surface getDist(vec3 p)
{
    // float ds = heartDist(p,vec3(0.0,1.0,6.0),1.0);
    // Surface Heart;
    // Heart.sd = ds; Heart.id=0;
    // Heart.color = vec3 (0.9,0.1,0.05); Heart.Ka=0.2; Heart.Kd=0.3;Heart.Ks=0.5;
    
    //Chão
    float dp = planeDist(p, vec3(0.0, 1.5, 0.0), 0.0);
    Surface Plane;
    Plane.sd = dp;
    Plane.id = 1;

    // TEXTURA wood.jpg no chão
    vec2 uv = p.xz * 0.25;  // Ajuste escala da textura
    uv = fract(uv);         // Repete textura em padrão infinito
    Plane.color = texture(iChannel1, uv).rgb;

    // ESCURECIMENTO DO AMBIENTE
    // Chão
    Plane.Ka = 0.03; Plane.Kd = 0.08; Plane.Ks = 0.1;

    // float dp = planeDist(p,vec3 (0.0,1.0,0.0),0.0);
    // Surface Plane;
    // Plane.sd=dp;Plane.id=1;
    // Plane.color=vec3(0.75); Plane.Ka=0.2; Plane.Kd=0.4;Plane.Ks=0.4;
    // Surface d= unionS(Heart,Plane);
    Surface d = Plane;
    // --- TAPETE VERMELHO CENTRAL ---
    // Posição central do tapete (ajuste Z para o centro do museu)
    vec3 centroTapete = vec3(0.0, -1.55, 0.0); // y=0.01 para ficar logo acima do chão

    // Elipse: raio maior em X, menor em Z
    vec2 raioTapete = vec2(5.0, 2.0); // ajuste o tamanho conforme desejar

    // SDF de elipse no plano XZ
    vec2 pTapete = (p - centroTapete).xy / raioTapete;
    float tapeteSD = length(pTapete) - 1.0;

    // Efeito de profundidade: gradiente radial
    float grad = clamp(1.0 - length(pTapete), 0.0, 1.0);
    vec3 corTapete = mix(vec3(0.3, 0.0, 0.0), vec3(1.0, 0.1, 0.1), pow(grad, 1.5));

    // Superfície do tapete
    Surface Tapete;
    Tapete.sd = tapeteSD;
    Tapete.color = corTapete;
    // ESCURECIMENTO DO AMBIENTE
    // Tapete
    Tapete.Ka = 0.12; Tapete.Kd = 0.18; Tapete.Ks = 0.07;
    Tapete.id = 99;

    // Adiciona o tapete ao cenário
    d = unionS(Tapete, d);
    
    // --- QUEIJO COM FATIA E ÁRVORE ---
    float t = iTime;
    vec3 deslocamentoQueijo = vec3(5.0, -0.40, 0.0);
    mat4 mQueijo = rotY(40.0 * t) * trans(deslocamentoQueijo);
    vec3 pQueijo = opTransf(p, mQueijo);

    // Cilindro principal do queijo
    float queijoSD = cylinderVerticalDist(pQueijo, vec2(1.7, 0.3));

    // Fatia (corte)
    float angle = radians(40.0);
    vec3 n1 = normalize(vec3(sin(angle), 0.0, -cos(angle)));
    vec3 n2 = normalize(vec3(-sin(angle), 0.0, -cos(angle)));
    float corte1 = planeDist(pQueijo, n1, 0.0);
    float corte2 = planeDist(pQueijo, n2, 0.0);
    float corte3 = -cylinderVerticalDist(pQueijo, vec2(2.0, 0.3));
    float fatia = max(max(corte1, corte2), corte3);

    // SDF final do queijo com fatia
    float queijoFinalSD = max(queijoSD, -fatia);

    Surface Queijo;
    Queijo.sd = queijoFinalSD;
    Queijo.color = vec3(1.2, 1.0, 0.3);
    Queijo.Ka = 0.3; Queijo.Kd = 0.5; Queijo.Ks = 0.2;
    Queijo.id = 2;
    d = unionS(Queijo, d);

    // --- ÁRVORE NO CENTRO DO QUEIJO ---
    vec3 centroArvore = deslocamentoQueijo + vec3(-10.0, 0.8, 0.0); // centro do queijo, em cima dele

    // Tronco (cilindro vertical, mais alto)
    vec3 pTronco = p - centroArvore;
    float troncoAltura = 1.1;
    float troncoSD = cylinderVerticalDist(pTronco, vec2(0.08, troncoAltura));

    // Copa principal (esfera maior em cima do tronco)
    vec3 copaPos = centroArvore + vec3(0.0, troncoAltura, 0.0);
    float copaSD = length(p - copaPos) - 0.32;

    // Coordenadas UV esféricas para textura
    vec3 dir = normalize(p - copaPos);
    float u = 0.5 + atan(dir.z, dir.x) / (2.0 * PI);
    float v = 0.5 - asin(dir.y) / PI;
    vec2 uvCopa = vec2(u, v);

    // Junta tronco e copa
    float arvoreSD = min(troncoSD, copaSD);

    Surface Arvore;
    Arvore.sd = arvoreSD;
    // Corrigido: aplica textura na copa usando máscara
    float copaMask = step(copaSD, troncoSD); // 1.0 se copaSD <= troncoSD
    Arvore.color = mix(vec3(0.4, 0.2, 0.05), texture(iChannel2, uvCopa).rgb, copaMask);
    Arvore.Ka = 0.3; Arvore.Kd = 0.6; Arvore.Ks = 0.2;
    Arvore.id = 60;

    d = unionS(Arvore, d);

    

    // Surface CSG;
    // CSG = csgObject(p); CSG.id=2;
    // d=unionS(CSG,d);

    //Esfera esquerda
    // Surface SphereLeft;
    // SphereLeft.sd = sphereDist(p-vec3( -3.0,1.0, 4.0),1.0);
    // SphereLeft.color = vec3(0.1,0.6,0.7); SphereLeft.Ka=0.2; SphereLeft.Kd=0.4;SphereLeft.Ks=0.4; SphereLeft.id=3;
    // d=unionS(SphereLeft,d);

    // Panela simples substituindo SphereLeft
    Surface panela;
    {
    // Parâmetros da rotação
    float angPanela = iTime * 0.7; // velocidade da rotação

    // Centro da panela (deslocado 2 unidades para a direita)
    vec3 centroPanela = vec3(-3.0, -1.75, 4.0) - vec3(2.0, 0.0, -2.0);

    // Aplica rotação no espaço da panela (em torno do centro)
    vec3 pPanela = p - centroPanela;
    pPanela.xz = rotate2d(angPanela) * pPanela.xz;
    pPanela += centroPanela;

    // Corpo
    vec3 pCorpo = pPanela - (vec3(-3.0, 0.8, 4.0) - vec3(2.0, 0.0, -2.0));
    float corpoSD = max(length(pCorpo.xz) - 1.0, abs(pCorpo.y) - 0.25);

    // Alça esquerda
    vec3 pHandleL = pPanela - (vec3(-2.0, 0.9, 4.0) - vec3(2.0, 0.0, -2.0));
    float alcaLSD = cylinderDist(pHandleL, vec3(0.0, 0.0, -0.3), vec3(0.0, 0.0, 0.3), 0.15);

    // Alça direita
    vec3 pHandleR = pPanela - (vec3(-4.0, 0.9, 4.0) - vec3(2.0, 0.0, -2.0));
    float alcaRSD = cylinderDist(pHandleR, vec3(0.0, 0.0, -0.3), vec3(0.0, 0.0, 0.3), 0.15);

    // Junta corpo e as duas alças
    float panelaSD = min(corpoSD, min(alcaLSD, alcaRSD));

    // --- TAMPA RETA ---
    // vec3 tampaPos = vec3(-3.0, 1.05, 4.0);
    // --- TAMPA ANIMADA (sobe e desce) ---
    float animTampa = 0.09 * sin(iTime * 2.0); // amplitude e velocidade
    vec3 tampaPos = (vec3(-3.0, 1.05 + animTampa, 4.0) - vec3(2.0, 0.0, -2.0));
    float tampaSD = cylinderVerticalDist(pPanela - tampaPos, vec2(1.0, 0.05));

    // --- SUPORTE DA TAMPA ---
    vec3 suportePos = tampaPos + vec3(0.0, 0.08, 0.0);
    float suporteSD = cylinderVerticalDist(pPanela - suportePos, vec2(0.13, 0.07));

    // Junta tampa e suporte
    float tampaCompletaSD = min(tampaSD, suporteSD);

    // Junta tudo com a panela
    float panelaCompletaSD = min(panelaSD, tampaCompletaSD);

    panela.sd = panelaCompletaSD;
    panela.color = vec3(0.7, 0.7, 0.7);
    panela.Ka = 0.3; panela.Kd = 0.6; panela.Ks = 0.8;
    panela.id = 10;
}
    d = unionS(panela, d);

    // --- PRATOS GIRANDO EM TORNO DA PANELA ---
    float yPrato = 0.8; // mesma altura do centro da panela
    vec3 centroPanelaPratos = vec3(-3.0, yPrato, 4.0) - vec3(2.0, 0.0, -2.0); // centro da panela deslocado

    float raioPrato = 2.5; // distância dos pratos ao centro da panela
    float angulo = iTime * 0.7; // velocidade da rotação

    // Prato 1
    vec3 prato1Pos = centroPanelaPratos + vec3(
        raioPrato * cos(angulo),
        0.0,
        raioPrato * sin(angulo)
    );

    // Prato 2 (oposto ao prato 1)
    vec3 prato2Pos = centroPanelaPratos + vec3(
        raioPrato * cos(angulo + PI),
        0.0,
        raioPrato * sin(angulo + PI)
    );

    float prato1SD = cylinderVerticalDist(p - prato1Pos, vec2(0.8, 0.03));
    Surface Prato1;
    Prato1.sd = prato1SD;
    Prato1.color = vec3(0.95, 0.95, 0.85);
    Prato1.Ka = 0.3; Prato1.Kd = 0.6; Prato1.Ks = 0.4;
    Prato1.id = 20;
    d = unionS(Prato1, d);

    float prato2SD = cylinderVerticalDist(p - prato2Pos, vec2(0.8, 0.03));
    Surface Prato2;
    Prato2.sd = prato2SD;
    Prato2.color = vec3(0.95, 0.95, 0.85);
    Prato2.Ka = 0.3; Prato2.Kd = 0.6; Prato2.Ks = 0.4;
    Prato2.id = 21;
    d = unionS(Prato2, d);


    // --- FRIGIDEIRA AO LADO DA PANELA ---
    // Deslocamento de 4 pixels para a direita
    vec3 deslocamentoFrigideira = vec3(5.0, 0.0, 1.0);
    vec3 centroFrigideira = vec3(-1.0, 1.5, 4.0) + deslocamentoFrigideira; // posição ao lado da panela

    // Corpo da frigideira (cilindro baixo)
    vec3 pFrigideira = p - centroFrigideira;
    float corpoFrigideiraSD = max(length(pFrigideira.xz) - 1.1, abs(pFrigideira.y) - 0.10);

    // Cabo (cilindro fino)
    vec3 pCabo = p - (centroFrigideira + vec3(1.0, 0.0, 0.0)); // deslocado para a direita
    float caboSD = cylinderDist(pCabo, vec3(0.0, 0.0, 0.0), vec3(1.2, 0.0, 0.0), 0.08);

    // Junta corpo e cabo
    float frigideiraSD = min(corpoFrigideiraSD, caboSD);

    // --- TAMPA DA FRIGIDEIRA ANIMADA (sobe e desce) ---
    float animTampaFrigideira = 0.05 * sin(iTime * 2.0 + 1.5); // fase diferente da panela
    vec3 tampaFrigideiraPos = centroFrigideira + vec3(0.0, 0.13 + animTampaFrigideira, 0.0);
    float tampaFrigideiraSD = cylinderVerticalDist(p - tampaFrigideiraPos, vec2(1.1, 0.05));

    // Suporte da tampa
    vec3 suporteFrigideiraPos = tampaFrigideiraPos + vec3(0.0, 0.07, 0.0);
    float suporteFrigideiraSD = cylinderVerticalDist(p - suporteFrigideiraPos, vec2(0.13, 0.07));

    // Junta tampa e suporte
    float tampaCompletaFrigideiraSD = min(tampaFrigideiraSD, suporteFrigideiraSD);

    // Junta tudo (frigideira + tampa)
    float frigideiraCompletaSD = min(frigideiraSD, tampaCompletaFrigideiraSD);

    Surface Frigideira;
    Frigideira.sd = frigideiraCompletaSD;
    Frigideira.color = vec3(0.2, 0.2, 0.2); // cor escura
    Frigideira.Ka = 0.3; Frigideira.Kd = 0.6; Frigideira.Ks = 0.7;
    Frigideira.id = 11;
    d = unionS(Frigideira, d);

    // --- PRATOS GIRANDO EM TORNO DA FRIGIDEIRA ---
    float yPratoF = centroFrigideira.y; // mesma altura da frigideira
    float raioPratoF = 2.0; // raio da órbita dos pratos
    float angPratoF = iTime * 0.7; // velocidade da rotação

    // Prato 1
    vec3 pratoF1Pos = centroFrigideira + vec3(
        raioPratoF * cos(angPratoF),
        0.0,
        raioPratoF * sin(angPratoF)
    );

    // Prato 2 (oposto ao prato 1)
    vec3 pratoF2Pos = centroFrigideira + vec3(
        raioPratoF * cos(angPratoF + PI),
        0.0,
        raioPratoF * sin(angPratoF + PI)
    );

    float pratoF1SD = cylinderVerticalDist(p - pratoF1Pos, vec2(0.8, 0.03));
    Surface PratoF1;
    PratoF1.sd = pratoF1SD;
    PratoF1.color = vec3(0.2, 0.2, 0.2);
    PratoF1.Ka = 0.3; PratoF1.Kd = 0.6; PratoF1.Ks = 0.4;
    PratoF1.id = 22;
    d = unionS(PratoF1, d);

    float pratoF2SD = cylinderVerticalDist(p - pratoF2Pos, vec2(0.8, 0.03));
    Surface PratoF2;
    PratoF2.sd = pratoF2SD;
    PratoF2.color = vec3(0.2, 0.2, 0.2);
    PratoF2.Ka = 0.3; PratoF2.Kd = 0.6; PratoF2.Ks = 0.4;
    PratoF2.id = 23;
    d = unionS(PratoF2, d);


    // Surface OctPrism;
    // OctPrism.sd = octogonPrismDist(p-vec3( 3.0,0.8,5.0), 0.7, 0.25); OctPrism.id=4;
    // OctPrism.color = vec3(0.4,0.2,0.7); OctPrism.Ka=0.2; OctPrism.Kd=0.7;OctPrism.Ks=0.1;
    // d=unionS(OctPrism,d);



    // Surface Sphere;
    // Sphere.id=5;
    // Sphere.sd=sphereDist(p-vec3(4.0,1.0,2.0),1.0);
    // Sphere.Ka=0.2;Sphere.Kd=0.4;Sphere.Ks=0.4;Sphere.id=5;
    // Sphere.color=vec3(0.,1.0,0.);
    // d=unionS(d,Sphere);

    // Por este:
    // --- Barco afundando ---
   // vec3 barcoPos = vec3(4.0, 1.0, 2.0);
   float alturaBarco = 1.0 + 0.3 * sin(iTime * 2.0); // sobe e desce com amplitude 0.3 e frequência 2
   vec3 barcoPos = vec3(5.5 + 1.0, alturaBarco, 2.0); // desloca 1 unidade para a direita (eixo X)
   vec3 pBarco = p - barcoPos;

    // Casco: cilindro horizontal
    float casco = cylinderDist(pBarco, vec3(-2.1, -0.5, 0.0), vec3(2.1, -0.5, 0.0), 1.0);
    //float casco = cylinderDist(pBarco, vec3(0.0, -1.0, 0.0), vec3(0.0, 1.0, 0.0), 0.6);


    // Corte superior para simular afundando
    float corteAgua = planeDist(pBarco, vec3(0.0, 1.0, 0.0), -0.30); // só parte de baixo do barco
    //float corteAgua = planeDist(pBarco, vec3(0.0, 0.0, 1.0), 0.1);  // ajustado para cortar pela frente

    float barcoSDF = max(casco, corteAgua);

    // Mastro (opcional)
    float mastro = cylinderVerticalDist(pBarco - vec3(0.0, 0., 0.0), vec2(0.12, 0.5));
    //float mastro = cylinderDist(pBarco - vec3(0.0, 0.0, 0.6), vec3(0.0, -0.3, 0.6), vec3(0.0, 1.0, 0.6), 0.1);

    barcoSDF = min(barcoSDF, mastro);

    Surface Barco;
    Barco.sd = barcoSDF;
    Barco.color = vec3(0.3, 0.2, 0.1); // marrom escuro
    Barco.Ka = 0.2; Barco.Kd = 0.5; Barco.Ks = 0.3;
    Barco.id = 5;
    d = unionS(d, Barco);

        // --- ÁGUA AZUL ONDULADA EMBAIXO DO BARCO ---
    vec3 aguaCentro = vec3(6.0 + 1.0, 0.1, 2.0); // desloca 1 unidade para a direita (eixo X)
    float raioAgua = 2.8; // raio da "poça" de água

    // Efeito de onda: desloca a superfície da água em Y
    float onda = 0.08 * sin(2.0 * PI * (p.x + p.z) * 0.3 + iTime * 0.7);

    // SDF: plano em y=0 + onda, limitado a um disco
    float aguaSD = max(
        p.y - aguaCentro.y - onda, // plano ondulado
        length((p - aguaCentro).xz) - raioAgua // disco
    );

    Surface Agua;
    Agua.sd = aguaSD;
    Agua.color = vec3(0.1, 0.4, 0.8); // azul da água
    Agua.Ka = 0.4; Agua.Kd = 0.7; Agua.Ks = 0.3;
    Agua.id = 80;
    d = unionS(Agua, d);
    // Cabine do capitão
    vec3 cabinePos = barcoPos + vec3(0.0, -0.03, 0.0);
    float cabineSD = boxDist(p - cabinePos, vec3(1.0, 0.3, 0.3));
    barcoSDF = min(barcoSDF, cabineSD);

    // Janela
    vec3 janelaPos = cabinePos + vec3(0.0, 0.0, -0.41);
    float janelaSD = boxDist(p - janelaPos, vec3(0.18, 0.12, 0.05));
    barcoSDF = max(barcoSDF, -janelaSD);

    // Superfície da cabine
    Surface Cabine;
    Cabine.sd = cabineSD;
    Cabine.color = vec3(0.85, 0.85, 0.95);
    Cabine.Ka = 0.3; Cabine.Kd = 0.6; Cabine.Ks = 0.3;
    Cabine.id = 51;
    d = unionS(Cabine, d);

    // Superfície da janela
    Surface Janela;
    Janela.sd = janelaSD;
    Janela.color = vec3(0.3, 0.5, 0.8);
    Janela.Ka = 0.2; Janela.Kd = 0.7; Janela.Ks = 0.5;
    Janela.id = 52;
    d = unionS(Janela, d);


// --- TRÊS MASTROS COM BANDEIRAS ONDULANDO ---
// Posição base dos mastros (onde estava o OctPrism)
    vec3 baseMastro = vec3(0.0, 0.0, 15.0);
    float alturaMastro = 5.5; // triplo da altura original

    for (int i = 0; i < 3; i++) {
        float dx = float(i - 1) * 6; // espaçamento dos mastros no eixo X
        vec3 mastroPos = baseMastro + vec3(dx, 0.0, 0.0); // alinhados no eixo X

        // Mastro: cilindro vertical, bem mais alto
        float mastroSD = cylinderVerticalDist(p - mastroPos, vec2(0.13, alturaMastro));
        Surface Mastro;
        Mastro.sd = mastroSD;
        Mastro.color = vec3(0.7, 0.7, 0.7);
        Mastro.Ka = 0.2; Mastro.Kd = 0.6; Mastro.Ks = 0.3;
        Mastro.id = 30 + i;
        d = unionS(Mastro, d);

        // Bandeira: faixa animada com vento, presa na ponta do mastro
        float largura = 2.0; // dobrado
        float altura = 1.0;  // dobrado
        // Posição da ponta presa ao mastro (canto esquerdo da bandeira)
        vec3 origem = mastroPos + vec3(-0.09, alturaMastro, -0.6);
        vec3 pBandeira = p - origem;
        // Move a origem para o canto da bandeira (em z = -largura)
        //pBandeira.z += largura; 
        // Rotaciona -70º para a direita em torno do eixo Y
        float angulo = radians(-70.0);
        mat2 rotY = mat2(cos(angulo), -sin(angulo), sin(angulo), cos(angulo));
        vec2 xz = rotY * pBandeira.xz;
        pBandeira.x = xz.x;
        pBandeira.z = xz.y;
        // Move de volta para o centro da bandeira
        pBandeira.z -= largura;
        float z = pBandeira.z;
        float y = pBandeira.y;
        float x = pBandeira.x - 0.10 * sin(2.0 * z + iTime * 1.2 + float(i) * 1.5);
        float bandeiraSD = max(max(abs(z) - largura, abs(y) - altura), abs(x) - 0.02);

        Surface Bandeira;
        Bandeira.sd = bandeiraSD;
        // --- TEXTURA NA BANDEIRA 0 ---
        if (i == 0) {
            float u = (z + largura) / (2.0 * largura);
            float v = 1.0 - (y + altura) / (2.0 * altura);
            vec2 uv = vec2(u, v);
            Bandeira.color = texture(iChannel4, uv).rgb * 0.8;
        }
        if (i == 1) {
            float u = (z + largura) / (2.0 * largura);
            float v = 1.0 - (y + altura) / (2.0 * altura);
            vec2 uv = vec2(u, v);
            Bandeira.color = texture(iChannel5, uv).rgb * 0.8;
        }
        if (i == 2) {
            float u = (z + largura) / (2.0 * largura);
            float v = 1.0 - (y + altura) / (2.0 * altura);
            vec2 uv = vec2(u, v);
            Bandeira.color = texture(iChannel6, uv).rgb * 0.8;
        }
        Bandeira.Ka = 0.3; Bandeira.Kd = 0.7; Bandeira.Ks = 0.4;
        Bandeira.id = 40 + i;
        d = unionS(Bandeira, d);
    }
        
// // --- PÓDIO DE 3 LUGARES ---
// // Posição base do pódio (mais para trás no eixo Z)
// vec3 podiumBase = vec3(0.0, 0.0, 10.0); // mais para trás

// // Alturas dos degraus (3x mais alto)
// float h1 = 0.7 * 3.0; // 1º lugar (mais alto)
// float h2 = 0.45 * 3.0; // 2º lugar
// float h3 = 0.3 * 3.0; // 3º lugar

// // Largura/profundidade dos degraus (5x mais largo)
// vec3 size1 = vec3(0.45 * 5.0, h1, 0.45 * 5.0);
// vec3 size2 = vec3(0.45 * 5.0, h2, 0.45 * 5.0);
// vec3 size3 = vec3(0.45 * 5.0, h3, 0.45 * 5.0);

// // 1º lugar (centro)
// vec3 pos1 = podiumBase + vec3(0.0, h1/2.0, 0.0);
// float podium1SD = boxDist(p - pos1, size1);

// // 2º lugar (esquerda)
// vec3 pos2 = podiumBase + vec3(-0.55 * 5.0, h2/2.0, 0.0);
// float podium2SD = boxDist(p - pos2, size2);

// // 3º lugar (direita)
// vec3 pos3 = podiumBase + vec3(0.55 * 5.0, h3/2.0, 0.0);
// float podium3SD = boxDist(p - pos3, size3);

// // Junta os três degraus
// float podiumSD = min(podium1SD, min(podium2SD, podium3SD));

// // Cores diferentes para cada degrau
// vec3 podiumColor = vec3(0.9); // padrão: branco
// if (podiumSD == podium1SD) podiumColor = vec3(1.0, 0.85, 0.2); // ouro
// if (podiumSD == podium2SD) podiumColor = vec3(0.7, 0.7, 0.7); // prata
// if (podiumSD == podium3SD) podiumColor = vec3(0.8, 0.5, 0.2); // bronze

// Surface Podium;
// Podium.sd = podiumSD;
// Podium.color = podiumColor;
// Podium.Ka = 0.3; Podium.Kd = 0.6; Podium.Ks = 0.3;
// Podium.id = 70;
// d = unionS(Podium, d);

    // --- PAREDE VERTICAL ATRÁS DOS MASTROS ---
    float paredeZ = 4.5; // valor maior que o Z dos mastros (ajuste conforme necessário)
    float paredeSD = planeDist(p, vec3(0.0, 0.0, -0.20), -paredeZ);

    Surface Parede;
    Parede.sd = paredeSD;
    Parede.color = vec3(0.05, 0.02, 0.01); // marrom mais escuro

    // ESCURECIMENTO DO AMBIENTE
    // Parede
    Parede.Ka = 0.02; Parede.Kd = 0.06; Parede.Ks = 0.05;

    Parede.id = 100;
    d = unionS(Parede, d);

    // --- QUADRO SEM MOLDURA NA FRENTE DA PAREDE ---
    float quadroY = 1.0; // altura do centro do quadro
    float quadroX = 0.01; // centralizado em X
    float quadroLargura = 4.0;
    float quadroAltura = 4.0;

    // Quadro (box, sem moldura)
    vec3 centroQuadro = vec3(quadroX, 3.0, 20.0); // levemente à frente da parede
    float quadroSD = boxDist(p - centroQuadro, vec3(quadroLargura/2.0, quadroAltura/2.0, 0.03));

    Surface Quadro;
    Quadro.sd = quadroSD;
    Quadro.color = vec3(0.95, 0.95, 0.85); // cor clara (ou troque por textura)

    // ESCURECIMENTO DO AMBIENTE
    // Quadro
    Quadro.Ka = 0.04; Quadro.Kd = 0.09; Quadro.Ks = 0.05;

    Quadro.id = 202;
    d = unionS(Quadro, d);
    return d;
}



Surface rayMarching(vec3 Cam, vec3 rd)
{
    float d0 = 0.0;
    vec3 pi = Cam +d0*rd;
    Surface dist = getDist(pi);
    int i=0;
    while ((i<MAX_STEPS)&&(dist.sd>SURF_DIST)&&(d0<MAX_DIST))
    {
        d0+=dist.sd;
        pi = Cam+d0*rd;
        dist=getDist(pi);
        i++;
    }
    if((i>MAX_STEPS)||(d0>MAX_DIST))
    {
        dist.color=Sky(rd);
        dist.sd=MAX_DIST;
    }
    else
        dist.sd=d0;
    return dist;
}

vec3 estimateNormal(vec3 p)
{
    float d= getDist(p).sd;
    float dx =  getDist(vec3(p.x+EPSILON,p.y,p.z)).sd-d;
    float dy = getDist(vec3(p.x,p.y+EPSILON,p.z)).sd-d;
    float dz = getDist(vec3(p.x-EPSILON,p.y,p.z+EPSILON)).sd-d;
    return normalize(vec3(dx,dy,dz));
}

mat3 setCamera(vec3 CamPos, vec3 LookAt)
{
   vec3 cd = normalize(LookAt-CamPos);
   vec3 cv = cross(cd,vec3(0.0,1.0,0.0));
   vec3 cu = cross(cv,cd);
   return mat3(-cv,cu,cd);
}

// https://iquilezles.org/articles/rmshadows
float calcSoftshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
    // bounding volume
    float tp = (tmax*0.75-ro.y)/rd.y; if( tp>0.0 ) tmax = min( tmax, tp );

    float res = 1.0;
    float t = mint;
    for( int i=0; i<24; i++ )
    {
                float h = getDist( ro + t*rd ).sd;
        float s = clamp(8.0*h/t,0.0,1.0);
        res = min( res, s );
        t += clamp( h, 0.01, 0.25 );
        if( res<0.001 || t>tmax ) break;
    }
    res = clamp( res, 0.0, 1.0 );
    return res*res*(3.0-2.0*res);
}

// "p" point apply texture to
// "n" normal at "p"
// "k" controls the sharpness of the blending in the
//     transitions areas.
// "s" texture sampler
vec3 boxmap( in sampler2D s, in vec3 p, in vec3 n, in float k )
{
    // project+fetch
        vec4 x = texture( s, p.yz );
        vec4 y = texture( s, p.zx );
        vec4 z = texture( s, p.xy );

    // and blend
    vec3 m = pow( abs(n), vec3(k) );
        return ((x*m.x + y*m.y + z*m.z) / (m.x + m.y + m.z)).xyz;
}

vec3 getLight(vec3 p,Surface s,vec3 Cam)
{
    if(s.sd==MAX_DIST)
        return s.color;

    vec3 LightColor = vec3(1.0);
    float exp=7.0;
    vec3 lightPos =vec3(2.0,4.0,2.0);

    vec3 LightColor1 = vec3(1.0,0.0,0.0);
    vec3 lightPos1 =vec3(-3.0,4.0,4.0);

    //lightPos.xz+=length(lightPos)*vec2(cos(iTime*0.75),sin(iTime*0.75));
    vec3 lightDir = normalize(lightPos-p);
    vec3 lightDir1 = normalize(lightPos1-p);
    vec3 eye = normalize(Cam-p);

    vec3 N = estimateNormal(p);
    vec3 R = normalize(reflect(-lightDir,N));
    vec3 R1 = normalize(reflect(-lightDir1,N));
    float l=dot(N,lightDir);
    l= clamp(l,0.0,1.0);
    float l1=dot(N,lightDir1);
    l1= clamp(l1,0.0,1.0);

    if(s.id == 60) {
        // Copa da árvore: não aplicar textura de queijo, usar apenas s.color
    }
    else if(s.id == 1) 
    {
        // Reaplica textura no plano para garantir iluminação correta
        vec2 uv = p.xz * 0.25;
        uv = fract(uv);
        s.color = texture(iChannel1, uv).rgb;
    }
    else if(s.id==3)
    {
        float theta = acos(N.z);
        float phi = PI+ atan(N.y/N.x);
        float x = theta/PI;
        float y =phi/(2.0*PI);
        s.color = mix(s.color,texture(iChannel0,vec2(x,y)).xyz,0.5);
    }
    else if(s.id==5)
    {
        float theta = acos(N.z);
        float phi = PI+ atan(N.y/N.x);
        float x = theta/PI;
        float y =phi/(2.0*PI);
        float n =perlin_noise(vec2(x,y)*iResolution.xy);
        s.color =vec3(abs(cos(n*10.0)));
    }
    else if(s.id == 6)
    {
        // Mapeamento procedural para efeito cortado
        s.color=boxmap(iChannel0,p,N,1.0);
        float grain = sin(30.0 * p.x) * cos(30.0 * p.z);
        grain *= 0.3 + 0.7 * perlin_noise(p.xz * 10.0);
        s.color = mix(s.color, vec3(1.0, 0.9, 0.5), grain);  // parte interna mais clara
    }
    else if(s.id==2)
    {
        s.color=boxmap(iChannel0,p,N,1.0);
    }

    if(s.id == 202) {
    // Mapeamento UV para o quadro (ajuste conforme posição/tamanho do quadro)
    // Supondo que o quadro está em centroQuadro, com largura e altura conhecidas:
    vec3 quadroCentro = vec3(0.03, -3.0, 20.0); // mesmo que no getDist
    float quadroLargura = 5.0;
    float quadroAltura = 5.0;
    vec2 uv = (p.xy - quadroCentro.xy) / vec2(quadroLargura, quadroAltura) + 0.5;
    uv.y = 1.0 - uv.y; // inverte o eixo Y
    s.color = texture(iChannel3, uv).rgb;
    }
    //phong contrib
    vec3 Is=vec3(0.);
   vec3 Is1=vec3(0.);
    float dotRN = dot(R,eye);
    if(dotRN>0.0)
    {
        Is=LightColor*s.Ks*pow(dotRN,exp);//*calcSoftshadow( p+10.0*EPSILON*R, R, 0.1, 3.0 );
    }

    float dotRN1 = dot(R1,eye);
    if(dotRN1>0.0)
    {
        Is1=LightColor1*s.Ks*pow(dotRN1,exp);//*calcSoftshadow( p+10.0*EPSILON*R1, R1, 0.1, 3.0 );
    }
   float  ss =calcSoftshadow( p+10.0*EPSILON*lightDir,lightDir, 0.1, 3.0 );
   float ss1=calcSoftshadow( p+10.0*EPSILON*lightDir1,lightDir1, 0.1, 3.0 );
    vec3 c = s.color*s.Ka+(s.color*l*s.Kd)*ss+(s.color*l1*s.Kd)*ss1+Is+ Is1;
    //Surface sh = rayMarching(p+100.0*EPSILON*lightDir,lightDir);
    //if(sh.sd<length(p-lightPos)) c*=0.2;
    //c*= ;
    return c;
}




void main()
{

vec2 p = (gl_FragCoord.xy-0.5*iResolution)/iResolution.y;


vec3 rd = normalize(vec3(p,1.5));
vec3 col;
// normalized mouse coordinates


    vec2 mo = (iMouse.xy )/iResolution.xy;
    vec3 Cam= vec3 (0.0,-3.0,0.5);
    float am = mix(-0.5*PI,0.5*PI,iMouse.z*mo.x);
    float bm = mix(-0.25*PI,0.25*PI,iMouse.z*mo.y);

    vec3 Target = vec3 (1.0,1.0,6.0);
    Cam.xz+=length(Cam-Target)*vec2(cos(am),sin(am));;
    Cam.yz+=length(Cam-Target)*vec2(cos(bm),sin(bm));;
    mat3 Ca = setCamera(Cam,Target);
    rd=Ca*rd;

Surface d = rayMarching(Cam,rd);
vec3 po= Cam+d.sd*rd;
vec3 l = getLight(po,d,Cam);

col = l;
// gamma
        col = pow( col, vec3(0.4545) );
C = vec4( col, 1.0 );

}
