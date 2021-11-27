DIR="$(dirname "$(realpath "$0")")"
cd $DIR

rm -rf ./_build
rm -rf ./api
rm -rf ../docs

mkdir ./api
mkdir ../docs
touch ../docs/.nojekyll
sphinx-apidoc -f -o ./api ../tensorslow
make html
mv _build/html/* ../docs
rm -rf _build