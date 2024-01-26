using CairoMakie

function get_theme()
    tw = 1.85
    ts = 10
    mts = 5
    ls = 15
    fontsize_theme = Theme(fontsize=25, Axis=(yticksmirrored=true,
            xticksmirrored=true,
            xminorticksvisible=true,
            yminorticksvisible=true,
            xminorgridvisible=false,
            xgridvisible=false,
            yminorgridvisible=false,
            ygridvisible=false,
            xminortickalign=1,
            yminortickalign=1,
            xtickalign=1,
            ytickalign=1, xticklabelsize=ls, yticklabelsize=ls,
            xticksize=ts, xtickwidth=tw,
            xminorticksize=mts, xminortickwidth=tw,
            yticksize=ts, ytickwidth=tw,
            yminorticksize=mts, yminortickwidth=tw), Lines=(
            linewidth=2.5,))
    return fontsize_theme
end