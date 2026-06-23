Token types:
Token type -> equivalent html tag.

- Page -> Whole page content `<body>`
- Title -> Main page title `<h1>`
- CatTitle -> Category title `<h2>`
- SubCatTitle -> Sub category title `<h3>` // Note: for later
- Paragraph -> `<p>`
- Schema -> `<img>` or `<svg>`

// -> optional
Ex in equivalent html tag:

```html
<body>
  //
  <h1>Some random course</h1>
  //
  <h2>The main category might be in another page</h2>
  <p>Some randoms things about the random course</p>
  <img src="./a_super_illustration_of_random_things.png" />
  <p>Some other randoms things about the random course</p>
  <h2>Another category</h2>
  <h3>Other category 1</h3>
</body>
```
Algorithm 

title is centered can only be the first line of the page, x1.2-1.5 scale of a normal line
then skip 2 lines
2/3 title chance

Select Type -> .1 cat title, .7 paragraph, .2 schema
cat title cant be directly after another cat title.

skip line - > cat titles -> x1.1-1.3 scale of normal line, max 3-5 words -> skip line

paragraph -> select nb lines 3-8 -> skip line

schema -> x0.6-.9 width of the page, keep aspect ratio. + random offset (0 ~ page_width - schema width)