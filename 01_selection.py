import pandas as pd
import numpy as np

from dotenv import load_dotenv
import os
load_dotenv()

from dl import authClient as ac, queryClient as qc
import time

token = ac.login(os.getenv('DATALAB_LOGIN'), os.getenv('DATALAB_PASSWORD'))
token

w = 2  # width of the strip in degrees
radius_gaia = 3.0 / 3600
radius_wise = 2.0 / 3600

for ra_mid in range(90 * w + w // 2, 270, w):
    print(f'Processing {ra_mid-w/2} to {ra_mid+w/2}...')
    file = f'./data/candidates/lists/candidates_{int(ra_mid-w/2):03d}-{int(ra_mid+w/2):03d}_2.csv'
    query = f"""
    SELECT
        -- SkyMapper Columns
        s.object_id as source_id_smss,
        s.raj2000 as ra_smss, s.dej2000 as dec_smss,
        s.glon, s.glat,
        s.g_psf, s.e_g_psf, s.g_petro, s.e_g_petro, 
        s.r_psf, s.e_r_psf, s.r_petro, s.e_r_petro,
        s.i_psf, s.e_i_psf, s.i_petro, s.e_i_petro,
        s.class_star, s.mean_fwhm,
        
        -- Gaia Columns
        g.source_id as source_id_gaia,
        ((s.raj2000 - g.ra) * (s.raj2000 - g.ra) + (s.dej2000 - g.dec) * (s.dej2000 - g.dec)) as dist_gaia2,
        g.ra as ra_gaia, g.dec as dec_gaia,
        g.phot_g_mean_mag, g.phot_bp_mean_mag, g.phot_rp_mean_mag,
        g.bp_rp, g.bp_g, g.g_rp, g.pseudocolour, g.pseudocolour_error,
        g.pm, g.pmra, g.pmdec, g.parallax,
        g.classprob_dsc_combmod_galaxy, g.classprob_dsc_combmod_quasar, g.classprob_dsc_combmod_star,
        
        -- WISE Columns
        w.source_id as source_id_wise,
        ((s.raj2000 - w.ra) * (s.raj2000 - w.ra) + (s.dej2000 - w.dec) * (s.dej2000 - w.dec)) as dist_wise2,
        w.ra as ra_wise, w.dec as dec_wise,
        w.w1mpro, w.w2mpro, w.w1mpro-w.w2mpro as w12mpro,
        w.w1mag, w.w2mag, w.w1mag-w.w2mag as w12mag

    FROM skymapper_dr4.master AS s

        -- Lateral Join with Gaia Sources
        LEFT JOIN LATERAL (
            SELECT gg.* 
            FROM gaia_dr3.gaia_source AS gg
            WHERE q3c_join(s.raj2000, s.dej2000, gg.ra, gg.dec, {radius_gaia})
            AND NOT (( (gg.pmra * gg.pmra) + (gg.pmdec * gg.pmdec) ) >= 
               9 * ( (gg.pmra_error * gg.pmra_error) + (gg.pmdec_error * gg.pmdec_error) )
                     AND gg.pm != 'nan')
        ) AS g ON TRUE

        -- Changed to INNER JOIN LATERAL for WISE Sources
        INNER JOIN LATERAL (
            SELECT cw.*
            FROM catwise2020.main AS cw
            WHERE 
                q3c_join(s.raj2000, s.dej2000, cw.ra, cw.dec, {radius_wise})
                AND (cw.w1mpro - cw.w2mpro) BETWEEN 0.4 AND 1.5
                AND (s.i_petro - cw.w1mpro) BETWEEN 2.6 AND 4.6
            -- LIMIT 1 -- Adjust as needed
        ) AS w ON TRUE

    WHERE 
        s.i_petro BETWEEN 17 AND 20
        -- ra range
        AND s.raj2000 BETWEEN {ra_mid - w / 2} AND {ra_mid + w / 2}

        AND ABS(s.glat) > 10
        -- Ensure at least two Gaia sources are associated
        AND 
        (
            SELECT COUNT(*)
            FROM gaia_dr3.gaia_source AS gg
            WHERE q3c_join(s.raj2000, s.dej2000, gg.ra, gg.dec, {radius_gaia})
            AND NOT (( (gg.pmra * gg.pmra) + (gg.pmdec * gg.pmdec) ) >= 
               9 * ( (gg.pmra_error * gg.pmra_error) + (gg.pmdec_error * gg.pmdec_error) )
                     AND gg.pm != 'nan')
        ) >= 2
        -- Exclude Major and Minor Magellanic Cloud Areas
        AND NOT (
            -- Exclude Large Magellanic Cloud (LMC) Area
            q3c_join(s.raj2000, s.dej2000, 80.894, -69.756, 6.0)
            OR
            -- Exclude Small Magellanic Cloud (SMC) Area
            q3c_join(s.raj2000, s.dej2000, 13.158, -72.822, 3.5)
        )
    LIMIT 1000;
    """

    if os .path.exists(file):
        print(f'File {file} exists. Skipping...')
        continue
    df = qc.query(sql=query, fmt='pandas', timeout=1200)
    df.to_csv(file, index=False)
    print(f'Processed {ra_mid-w/2} to {ra_mid+w/2}, {len(df)} candidates found. Time: {time.ctime()}')
    # break
