import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy_healpix import HEALPix
import astropy.units as u
from scipy.interpolate import griddata

def plot_density_mollweide(field, order="nested", title="Density map",
                           nlon=720, nlat=360):
    nside = int(np.sqrt(len(field) / 12))
    hp = HEALPix(nside=nside, order=order)
    lon, lat = hp.healpix_to_lonlat(np.arange(hp.npix))
    lon = lon.to_value(u.rad)
    lat = lat.to_value(u.rad)
    lon = ((lon + np.pi) % (2*np.pi)) - np.pi  # wrap to [-π, π]

    # Regular grid (edges for pcolormesh)
    lon_edges = np.linspace(-np.pi, np.pi, nlon)
    lat_edges = np.linspace(-np.pi/2, np.pi/2, nlat)
    Lon, Lat = np.meshgrid(lon_edges, lat_edges)

    # Interpolate field to regular grid
    vals = griddata(
        points=np.column_stack((lon, lat)),
        values=field,
        xi=np.column_stack((Lon.ravel(), Lat.ravel())),
        method="linear",
        fill_value=np.nan
    ).reshape(Lat.shape)

    vals = np.nan_to_num(vals, nan=0)

    fig = plt.figure(figsize=(9, 4.5))
    ax = fig.add_subplot(111, projection="mollweide")

    # use pcolormesh, which respects lon/lat orientation explicitly
    pcm = ax.pcolormesh(Lon, Lat, vals,
                        cmap="viridis",
                        shading="auto")
    plt.colorbar(pcm, orientation="horizontal", pad=0.05, label="Density")
    ax.set_title(title)
    ax.grid(True)
    plt.show()

    return fig


def plot_mollweide_plotly(field, order="nested", nlon=720, nlat=360):
    """
    Plot a HEALPix scalar field in a 2-D Mollweide projection using Plotly.
    """

    nside = np.sqrt(len(field) / 12).astype(int)

    hp = HEALPix(nside=nside, order=order)
    npix = hp.npix

    # Pixel centers
    lon, lat = hp.healpix_to_lonlat(np.arange(npix))
    lon = lon.to_value(u.rad)
    lat = lat.to_value(u.rad)

    # Build a regular lon/lat grid
    lon_grid = np.linspace(-np.pi, np.pi, nlon)
    lat_grid = np.linspace(-np.pi/2, np.pi/2, nlat)
    LON, LAT = np.meshgrid(lon_grid, lat_grid)

    # Interpolate the irregular healpix samples onto the lon/lat grid
    from scipy.interpolate import griddata
    vals = griddata(
        points=np.column_stack((lon, lat)),
        values=field,
        xi=np.column_stack((LON.ravel(), LAT.ravel())),
        method="linear",
        fill_value=np.nan,
    )
    img = vals.reshape(LAT.shape)

    fig = px.imshow(
        img,
        origin="lower",
        color_continuous_scale="Viridis",
        aspect="auto",
        title="Mollweide projection",
    )
    fig.update_xaxes(title="Longitude", showticklabels=False)
    fig.update_yaxes(title="Latitude", showticklabels=False)

    return fig

def plot_field_plotly(field, order='nested', r=1.0):

    nside = np.sqrt(len(field) / 12).astype(int)

    hp = HEALPix(nside=nside, order=order)
    lon, lat = hp.healpix_to_lonlat(np.arange(hp.npix))
    lon = lon.to_value(u.rad)
    lat = lat.to_value(u.rad)

    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)

    fig = go.Figure(data=[
        go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=2, color=field, colorscale='Viridis', showscale=True),
        )
    ])
    fig.update_layout(scene=dict(xaxis=dict(visible=False),
                                 yaxis=dict(visible=False),
                                 zaxis=dict(visible=False)))
    return fig
