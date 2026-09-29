# Setting up uploads to Cloudflare R2

Diagnostic bundles go straight to a Cloudflare R2 bucket. You do this once.

## 1. Create the bucket
1. Log in at <https://dash.cloudflare.com>.
2. In the left sidebar choose **Storage & databases > R2 object storage** (labelled **R2** in older dashboards). Add a payment method if asked; the free tier is generous.
3. Click **Create bucket**.
4. Name it `understudy-diagnostics`, leave the defaults (Automatic location, Standard), and click **Create bucket**. Keep it private; do not enable public access.

## 2. Create an API token
1. Go back to the R2 overview page and click **API** (or **Manage API tokens**) on the right.
2. Click **Create API token**.
3. Name it `understudy-laptop`.
4. Permissions: **Object Read & Write**.
5. Under **Specify bucket(s)**, choose **Apply to specific buckets only** and select `understudy-diagnostics`.
6. Leave TTL as forever (or set an expiry you will remember) and click **Create API Token**.

## 3. Copy the three values
The next page shows them once only:
- **Access Key ID**
- **Secret Access Key**
- the **Account ID**, shown on the R2 overview page (right side) and inside the endpoint URL `https://<account_id>.r2.cloudflarestorage.com`.

Ignore the "Token value". Keep the values somewhere safe until step 4 is done, then close the page.

## 4. Configure the laptop
Mac/Linux:

    ./bin/understudy upload --configure

Windows:

    bin\understudy.cmd upload --configure

Paste the Account ID, bucket name, Access Key ID and Secret Access Key (the secret is hidden as you type). The tool uploads and deletes a tiny test object to confirm it works, then saves the values to `~/.understudy/upload.toml` (readable only by you). Alternatively set `UNDERSTUDY_R2_ACCOUNT_ID`, `UNDERSTUDY_R2_BUCKET`, `UNDERSTUDY_R2_ACCESS_KEY_ID` and `UNDERSTUDY_R2_SECRET_ACCESS_KEY`.

## 5. Verify

    ./bin/understudy upload <session-folder> --dry-run

lists every file and size that would be sent, without sending. Then run it without `--dry-run`. Options: `--no-audio`, `--no-frames`, `--from SECONDS --to SECONDS`. Files already uploaded are skipped, so an interrupted upload can simply be re-run.

## 6. View uploads
Dashboard > R2 > `understudy-diagnostics` > **Objects**. Files live under `bundles/<computer-name>/<session>/<slice>/`. The `index.json` in that folder is written last, so its presence means the upload finished.

## Security
Anyone holding this key can write to (and read from) the bucket. It is scoped to that one bucket only. If a laptop is lost or stolen, go to R2 > API, find the token, **Roll** or **Delete** it, then create a new one for the remaining laptops.
