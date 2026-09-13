"""Emit an apply_patch patch for frame targets and a complete linked index."""
from pathlib import Path
import re, difflib, sys, math
sys.stdout.reconfigure(encoding='utf-8')
root=Path(__file__).resolve().parents[1]
changes={}; sections={}; state={'section':0,'frame':0}
def argument(s,pos):
    assert s[pos]=='{'
    depth=1; end=pos+1
    while depth:
        if s[end]=='{': depth+=1
        if s[end]=='}': depth-=1
        end+=1
    return s[pos+1:end-1],end
def frame_title(f):
    if r'\titlepage' in f: return 'Portada'
    m=re.search(r'\\frametitle\s*\{',f)
    if m: return argument(f,m.end()-1)[0]
    m=re.match(r'\\begin\{frame\}(?:\[[^\]]*\])?\s*\{',f)
    assert m,f[:120]
    return argument(f,m.end()-1)[0]
totals={'Lie':6,'Variacion':6,'Orden':3,'Holografia':6,'Escalar':6,'EQT':6,'HolografiaGen':6,'Discusion':4}
def process(relative):
    path=root/relative; old=path.read_text(encoding='utf-8-sig')
    if relative=='main.tex':
        offset=old.index(r'\begin{document}')+len(r'\begin{document}')
    else: offset=0
    body=old[offset:]
    pattern=r'\\begin\{frame\}(?:\[[^\]]*\])?[\s\S]*?\\end\{frame\}|\\section\{|\\Mapa(\w+)\{(\d+)\}|\\input\{([^}]+)\}'
    def replace(m):
        token=m.group()
        if token.startswith(r'\section'):
            title,_=argument(body,m.end()-1)
            state['section']+=1
            sections[state['section']]={'title':title,'items':[]}
        elif token.startswith(r'\input'):
            p=m[3]
            if p.startswith('contenido/') and not p.startswith('contenido/navegacion/'):
                process(p)
        elif token.startswith(r'\Mapa'):
            if int(m[2]) in (0,totals[m[1]]):
                title='Mapa de apertura' if int(m[2])==0 else 'Mapa de cierre'
                sections[state['section']]['items'].append((title,f"nav-map-{state['section']}-{m[2]}"))
        elif token.startswith(r'\begin{frame}'):
            title=frame_title(token)
            if title=='Portada': return token
            state['frame']+=1
            existing=re.search(r'\[label=([^\]]+)\]',token)
            target=existing[1] if existing else f"nav-unified-{state['frame']:03}"
            if not existing:
                token=re.sub(r'^(\\begin\{frame\})(?:\[[^\]]*\])?',lambda x:x[1]+'[label='+target+']',token,count=1)
            if state['section']:
                sections[state['section']]['items'].append((title,target))
            else:
                sections.setdefault(0,{'title':'Introducción','items':[]})['items'].append((title,target))
            return token
        return token
    new=old[:offset]+re.sub(pattern,replace,body)
    if old!=new: changes[relative]=(old,new)
process('main.tex')
assert state['section']==8
summary={
1:('Derivada de Lie e identidad principal','Dos cálculos, simetrías y controles EH y cosmológico.'),
2:('Variación de la acción','Palatini, volumen y frontera, Bianchi, corriente y potencial de Noether.'),
3:('Condición de segundo orden','Derivadas superiores y condición de Lanczos--Lovelock.'),
4:('Estructura holográfica LL','Datos de borde, separación de la acción y deducción holográfica.'),
5:('Generalización con campo escalar','Momentos, variación, Bianchi, Noether y desplazamiento.'),
6:('Estudio de casos: EQT',r'Sector cinético, reconstrucción de $L_0$ y acoplamiento $Q_\beta$.'),
7:('Holografía generalizada',r'Homogeneidad, defecto, superficie efectiva y controles de $Q_\beta$.'),
8:('Discusión de resultados',r'Divergencia de $P_\beta$, ansatz, límites y balance del modelo.')}
overview=[r'% Índice generado a partir de los frames; títulos de sección y accesos detallados.',r'\begin{frame}[label=nav-content]{Contenido}',r'\footnotesize',r'\begin{columns}[T,onlytextwidth]',r'\begin{column}{.48\textwidth}']
for n in range(1,9):
    if n==5: overview += [r'\end{column}',r'\begin{column}{.48\textwidth}']
    title,desc=summary[n]
    overview += [rf'\hyperlink{{nav-map-{n}-0}}{{\textbf{{{n}. {title}}}}}\par',desc+r'\par',rf'{{\scriptsize\hyperlink{{nav-index-{n}-1}}{{Ver todos los subtemas}}}}\par\medskip']
overview += [r'\end{column}',r'\end{columns}',r'\vfill',r'{\scriptsize Título: inicio de sección. Subtemas: acceso directo a cada diapositiva.\par',r'\hyperlink{nav-index-0-1}{Introducción}\qquad\hyperlink{'+sections[8]['items'][-1][1]+r'}{Referencias}}',r'\end{frame}']
detail=[r'\newcommand{\IndexEntry}[2]{%',r'\noindent\parbox[t]{.89\linewidth}{\raggedright\hyperlink{#1}{#2}}%',r'\hfill\parbox[t]{.08\linewidth}{\raggedleft\scriptsize\hyperlink{#1}{\pageref{#1}}}%',r'\par\medskip}', '']
for n,section in sorted(sections.items()):
    count=math.ceil(len(section['items'])/14)
    chunk=math.ceil(len(section['items'])/count)
    for k in range(count):
        items=section['items'][k*chunk:(k+1)*chunk]
        detail += [rf'\begin{{frame}}[label=nav-index-{n}-{k+1}]'+'{Índice: '+section['title']+'}',r'\footnotesize\raggedright',rf'\textcolor{{structure.fg}}{{Subtemas y diapositivas ({k+1}/{count})}}\par\medskip',r'\begin{columns}[T,onlytextwidth]']
        cut=math.ceil(len(items)/2)
        for col in [items[:cut],items[cut:]]:
            detail += [r'\begin{column}{.48\textwidth}']
            for title,target in col:
                detail += [rf'\IndexEntry{{{target}}}{{{title}}}']
            detail += [r'\end{column}']
        detail += [r'\end{columns}',r'\vfill']
        if k: detail += [rf'\hyperlink{{nav-index-{n}-{k}}}{{\scriptsize Anterior}}\qquad']
        if k+1<count: detail += [rf'\hyperlink{{nav-index-{n}-{k+2}}}{{\scriptsize Más subtemas}}\qquad']
        if n: detail += [rf'\hyperlink{{nav-map-{n}-0}}{{\scriptsize Inicio de sección}}']
        detail += [r'\end{frame}','']
for relative,lines in [('contenido/navegacion/contenido.tex',overview),('contenido/navegacion/indices.tex',detail)]:
    p=root/relative; changes[relative]=(p.read_text(encoding='utf-8') if p.exists() else None,'\n'.join(lines)+'\n')
changes = {p: pair for p, pair in changes.items()
           if pair[0] is None or pair[0].rstrip() != pair[1].rstrip()}
if '--check' in sys.argv:
    print('Índices actualizados.' if not changes else 'Requieren actualización: ' + ', '.join(changes))
    sys.exit(bool(changes))
if '--write' in sys.argv:
    for relative, (_, new) in changes.items():
        (root/relative).write_text(new, encoding='utf-8')
    print(f'Archivos actualizados: {len(changes)}')
    sys.exit(0)
print('*** Begin Patch')
for relative,(old,new) in changes.items():
    if old is None:
        print('*** Add File: '+str(root/relative).replace('\\','/'))
        print('\n'.join('+'+x for x in new.splitlines()))
    else:
        print('*** Update File: '+str(root/relative).replace('\\','/'))
        diff=list(difflib.unified_diff(old.splitlines(),new.splitlines(),n=2))[2:]
        print('\n'.join('@@' if x.startswith('@@') else x for x in diff))
print('*** End Patch')
