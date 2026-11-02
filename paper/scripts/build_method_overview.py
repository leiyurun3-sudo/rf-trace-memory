"""Update only figures/method_overview.pdf; no other figure or runtime writes.

Vector method diagram, 7.0 x 5.8 inches, minimum text size 8.6 pt.
Request-response protection and direct local correction define CORR25.
Named reference/current states separate capture from correction.
"""
from pathlib import Path
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'figures' / 'method_overview.pdf'
INK = '#202B33'
EDGE = '#9BA7B0'
BLUE = '#E9F1F7'
GOLD = '#FAF1DF'
GREEN = '#EAF3EF'
PURPLE = '#F1EBF4'


def build_method_overview():
    style = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
             'font.size': 8.6, 'mathtext.fontset': 'dejavusans',
             'pdf.fonttype': 42, 'ps.fonttype': 42, 'text.color': INK}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(7.0, 5.8), facecolor='white')
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set(xlim=(0, 700), ylim=(0, 580))
        ax.axis('off')

        def text(x, y, value, size=8.6, weight='normal', ha='left', color=INK):
            return ax.text(x, y, value, fontsize=size, fontweight=weight,
                           ha=ha, va='center', color=color)

        def card(x, y, w, h, title, fill):
            ax.add_patch(Rectangle((x, y), w, h, facecolor='white', edgecolor=EDGE,
                                   linewidth=.65, zorder=1))
            ax.add_patch(Rectangle((x+.5, y+h-27), w-1, 26.5,
                                   facecolor=fill, edgecolor='none', zorder=1))
            text(x+11, y+h-14, title, size=9.2, weight='bold')

        def arrow(start, end, color='#60717D'):
            ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>',
                                        mutation_scale=8, color=color, linewidth=.85,
                                        shrinkA=1.5, shrinkB=1.5, zorder=2))

        text(18, 565, 'TraceMemory: support, correction magnitude, and source memory',
             size=10.1, weight='bold')

        card(18, 494, 664, 54, 'Fixed source reference', BLUE)
        text(350, 507,
             r'Source image $x$ → controlled inversion → $r_{0:T}$; '
             r'$p_i=(r_{i+1}-r_i)/h_i$', ha='center')

        card(18, 306, 320, 168, '(a) Request-conditioned protection P', GOLD)
        text(30, 432, r'Source and target prompts; no evaluation-mask input')
        text(30, 415, 'Request response: 3 fixed nodes, 6 native calls')
        text(30, 398, r'$\sigma$ targets: $\{0.25,0.50,0.75\}$')
        # The probe uses native f_theta with guidance 3.5, not controlled u.
        # k indexes fixed reference nodes; j is spatial and ell is channel.
        text(30, 381, r'$\Delta_{k,j,\ell}=[f_\theta(r_k,\sigma_k,q_t,3.5)$')
        text(30, 364, r'$\qquad-f_\theta(r_k,\sigma_k,q_s,3.5)]_{j,\ell}$')
        text(30, 347, r'Channel RMS → q10/q90 clip → mean over $k$')
        text(30, 330, r'$P=1-0.75E\quad\in[0.25,1]$')
        text(30, 313, r'Spatial index $j$; probe node $k$; channel $\ell$', size=8.6)

        card(362, 306, 320, 168, '(b) Source residual at current state', BLUE)
        text(374, 432, r'Current edited state $z_i$; source prompt $q_s$')
        text(374, 407, r'Native source velocity: $u_i^s(z_i)$', size=9.0)
        text(374, 379, r'$c_i=p_i-u_i^s(z_i)$', size=10.0)
        text(374, 352, r'Reference motion $p_i$ comes from fixed $r_{0:T}$')
        text(374, 330, r'Source capture uses $r_i$; correction uses $z_i$')
        text(374, 314, 'Source evaluation is performed with memory off')

        card(18, 190, 664, 96, '(c) Direct correction and same-state energy control', GOLD)
        text(175, 243, 'Primary: TraceMemory (CORR25)', size=8.6,
             weight='bold', ha='center')
        text(175, 220, r'$\widehat c_i=0.25P\odot c_i$',
             size=9.6, ha='center')
        text(520, 243, 'Control: global residual direction', size=8.6,
             weight='bold', ha='center')
        text(520, 220, r'$\widehat c_i=0.25\,\frac{\mathrm{RMS}(P\odot c_i)}{\mathrm{RMS}(c_i)}c_i$', size=9.6, ha='center')
        text(350, 199, 'Equal correction RMS at a common state; independently evolving trajectories', ha='center')

        card(18, 15, 320, 155, '(d) Paired source K/V memory', PURPLE)
        text(30, 128, r'Source-prompt capture at reference state $r_i$')
        text(30, 108, r'Paired $(K^s_{\rm img},V^s_{\rm img})$; full-image source pool')
        text(30, 87, r'$R_s=\mathrm{softmax}(Q_tK_{\rm img}^{s\top}/\sqrt{d_h})V^s_{\rm img}$')
        text(30, 65, r'$O_{\rm memory}=O_{\rm text}+m_{\rm img}\odot R_s$')
        text(30, 44, 'Row-broadcast image mass; native text retained')
        text(30, 25, r'CORR25: all queries, $g_i=1$ for steps 0–6')

        card(362, 15, 320, 155, '(e) Current target: every spatial region', GREEN)
        text(374, 128, r'Target prompt $q_t$ at current edited state $z_i$')
        text(374, 108, r'$O_{\rm gate}=O_{\rm native}+g_iM_q$')
        text(374, 91, r'$\qquad\odot(O_{\rm memory}-O_{\rm native})$')
        text(374, 72, r'$z_{i+1}=z_i+h_i[\,u_i^{t,M}(z_i)+\widehat c_i\,]$', size=9.2)
        text(374, 53, 'Target velocity and integration remain global')
        text(374, 37, 'P routes the added correction only')
        text(374, 21, r'Final state $z_T$ → VAE decode → edited image')

        # Orthogonal adjacent-module connectors; named r_i avoids a crossing
        # long edge to source capture. No connector passes through a card.
        arrow((178, 494), (178, 474))
        arrow((522, 494), (522, 474))
        arrow((178, 306), (178, 286))
        arrow((522, 306), (522, 286))
        arrow((522, 190), (522, 170))
        arrow((338, 91), (362, 91))

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT, facecolor='white', metadata={
            'Title': 'TraceMemory method overview',
            'Subject': 'Request support; direct local correction and same-state energy control; reference source capture; current-state correction and global target update.',
            'Creator': 'build_method_overview.py'})
        fig.savefig(OUTPUT.with_suffix('.svg'), facecolor='white')
        fig.savefig(OUTPUT.with_suffix('.png'), facecolor='white', dpi=180)
        plt.close(fig)
    print(OUTPUT)


if __name__ == '__main__':
    build_method_overview()
