# Semantic Retrieval Review

These are retrieved source passages, not generated answers.
Up to 24 semantic matches and 24 keyword matches are combined, deduplicated, and reranked to select four passages.
Rerank scores are higher for stronger matches, but are not accuracy or confidence percentages.
Scores alone do not prove that a passage supports the requested answer.
For ranking, Day 1 is also expressed as first day, and similarly for other numbered days. Original source text below is unchanged.
The semantic-only report, when present, is preserved in retrieval_results_baseline.json.

The seven Singapore questions cover six PDF topics: indoor and outdoor activities are tested separately.
The final question demonstrates that an unsupported question can still retrieve passages.

Review the actual text against each question. Do not mark a case successful just because results exist.
Live weather and exchange rates will be handled by the custom MCP tools.

## 1. Major attractions and neighbourhoods

**Question:** Which neighbourhoods in Singapore can I explore for culture and sightseeing?

**What to check:** Named Singapore neighbourhoods and descriptions of their sights or cultural experiences.

**Review:** Pending. Mark useful / partly useful / unsupported, and explain why.

### Passage 1

**Source:** [Things To Do & Must-Visit Places in Singapore](https://www.visitsingapore.com/things-to-do/top-things-to-do/)

**Section:** Chinatown

**Rerank score:** 5.0006

**Found by:** semantic, keyword

**Chunk ID:** `singapore_things_to_do-016-000-fe67d0f0a80b`

> Destination: Singapore
> Section: Chinatown
> 
> Embrace Chinese heritage through bustling streets, vibrant culture and delectable cuisine in Chinatown.

### Passage 2

**Source:** [Singapore – Travel guide at Wikivoyage](https://en.wikivoyage.org/wiki/Singapore)

**Section:** See

**Rerank score:** 4.5121

**Found by:** semantic, keyword

**Chunk ID:** `singapore_wikivoyage-040-000-4a0f4acf1aed`

> Destination: Singapore
> Section: See
> 
> Sights in Singapore are covered in more detail under the various districts. Broadly speaking:  
> - Beaches and tourist resorts : Head to one of the three beaches on Sentosa or its southern islands. Other beaches can be found on the East Coast .  
> - Culture and cuisine : See Chinatown for Chinese treats, Little India for Indian flavours, Geylang Serai for a Malay experience or the East Coast for Eurasian and Peranakan culture and delicious seafood, including the famous chilli and black pepper crab.  
> - History and museums : The Bras Basah area east of Orchard and north of the Singapore River is Singapore's colonial core, with historical buildings and museums. All government-run museums in Singapore are free admission for Singapore citizens and permanent residents except for temporary exhibits, but visitors will be charged an admission fee.

### Passage 3

**Source:** [Things To Do & Must-Visit Places in Singapore](https://www.visitsingapore.com/things-to-do/top-things-to-do/)

**Section:** Katong-Joo Chiat

**Rerank score:** 3.9223

**Found by:** semantic

**Chunk ID:** `singapore_things_to_do-038-000-801c4064b4a3`

> Destination: Singapore
> Section: Katong-Joo Chiat
> 
> Discover the enchanting blend of Peranakan heritage, charming streets, and boutiques in the Katong-Joo Chiat neighborhood.

### Passage 4

**Source:** [Things To Do & Must-Visit Places in Singapore](https://www.visitsingapore.com/things-to-do/top-things-to-do/)

**Section:** Kampong Gelam

**Rerank score:** 3.8899

**Found by:** semantic, keyword

**Chunk ID:** `singapore_things_to_do-037-000-ab30c67d8604`

> Destination: Singapore
> Section: Kampong Gelam
> 
> Explore rich history, living traditions, excellent eateries and trendy shops in this vibrant neighborhood.

## 2. Local transportation

**Question:** How can I get around Singapore using the MRT and buses?

**What to check:** Useful information about Singapore's MRT or bus services.

**Review:** Pending. Mark useful / partly useful / unsupported, and explain why.

### Passage 1

**Source:** [Singapore – Travel guide at Wikivoyage](https://en.wikivoyage.org/wiki/Singapore)

**Section:** Get around > By bus

**Rerank score:** 8.3319

**Found by:** semantic, keyword

**Chunk ID:** `singapore_wikivoyage-034-000-5ab9459611ed`

> Destination: Singapore
> Section: Get around > By bus
> 
> Buses connect various corners of Singapore, but are slower and harder to use than the MRT. On a long-distance bus, frequent stops and slow speeds may mean your journey could take two to three times as long as the same trip via MRT. Their main advantage is you get to see the sights rather than a dark underground tunnel, partly if you can snag a front seat on a double-decker, which make up a relatively huge percentage of buses in Singapore. Moreover, since there are a lot more bus stops than MRT stations, buses can often get you closer to your destination, which makes a big difference in the equatorial heat and humidity. All public buses in Singapore are air conditioned and wheelchair accessible.

### Passage 2

**Source:** [Singapore Travel Guide & Tips | Travel Essentials](https://www.visitsingapore.com/travel-tips/essential-travel-information/)

**Section:** Public Transport

**Rerank score:** 7.1771

**Found by:** semantic

**Chunk ID:** `singapore_essential_information-016-000-b86667c10c7e`

> Destination: Singapore
> Section: Public Transport
> 
> Singapore's public transport system is fast and efficient. The MRT (Mass Rapid Transit) and bus systems have an extensive network of routes that will help you zip around the city. Plan your journey with OneMap ( App Store / Play Store ) to find accessible routes, nearby amenities and convenient ways to get around Singapore.

### Passage 3

**Source:** [Singapore – Travel guide at Wikivoyage](https://en.wikivoyage.org/wiki/Singapore)

**Section:** Get around > By bus

**Rerank score:** 6.2773

**Found by:** semantic, keyword

**Chunk ID:** `singapore_wikivoyage-034-001-612c7cf1f672`

> Destination: Singapore
> Section: Get around > By bus
> 
> Buses in Singapore stop on request only . Flag the bus if you want to board, press the stop button if you want to alight. All buses are GPS equipped; use an app like MyTransportSG , Google Maps, or Citymapper to track live arrival times. Most bus services have a frequency of between 5 and 15 minutes, although express services (look for bus numbers starting with a 5 or 6, or suffixed with "e"), have a frequency of 20 minutes or more and charge a higher fare.  
> The four bus operators in Singapore are SBS Transit, SMRT Buses, Tower Transit and Go-Ahead.

### Passage 4

**Source:** [Singapore – Travel guide at Wikivoyage](https://en.wikivoyage.org/wiki/Singapore)

**Section:** Get around > Fares

**Rerank score:** 6.1233

**Found by:** semantic, keyword

**Chunk ID:** `singapore_wikivoyage-031-002-1c3281e9d813`

> Destination: Singapore
> Section: Get around > Fares
> 
> Cash is not accepted on the MRT/LRT. More expensive cash fares are available on buses but no change is given. Children under seven years old travel free. If your child is above 0.9m in height they need a Child Concession Card to travel.  
> Remember to tap your device or card against a reader before you exit a bus. Otherwise, you will be charged the maximum fare. The bus operator has no idea where you got off and assumes that you rode to the end of the line.

## 3. Cultural and practical travel tips

**Question:** What should visitors know about dress and shoes when visiting temples in Singapore?

**What to check:** Visitor etiquette concerning clothing or removing shoes at places of worship.

**Review:** Pending. Mark useful / partly useful / unsupported, and explain why.

### Passage 1

**Source:** [Singapore – Travel guide at Wikivoyage](https://en.wikivoyage.org/wiki/Singapore)

**Section:** Respect

**Rerank score:** 1.4408

**Found by:** semantic, keyword

**Chunk ID:** `singapore_wikivoyage-095-001-6b135ca9bb4a`

> Destination: Singapore
> Section: Respect
> 
> That said, Singaporeans tend to be more socially conservative than Westerners, meaning that public display of affection is still frowned upon: holding hands is fine, but making out in public is considered to be impolite. Toplessness for women is not acceptable anywhere, even on the beach. Most places of worship require visitors to be conservatively dressed - no bare shoulders or midriff, and no shorts or skirts above the knee-cap. The major touristy places of worship will have shawls and sarongs so visitors can cover up before entering. Many places of worship also require you to remove your shoes before you enter.  
> The local dialect with its heavy Chinese influences may appear brusque or even rude, but saying "You want beer or not?" is in fact more polite in Chinese than asking if you want beer; after all, the person asking you the question is offering you a choice, not making a demand.

### Passage 2

**Source:** [Singapore – Travel guide at Wikivoyage](https://en.wikivoyage.org/wiki/Singapore)

**Section:** Respect > Visiting homes

**Rerank score:** 0.4904

**Found by:** semantic, keyword

**Chunk ID:** `singapore_wikivoyage-097-000-c8e95883274f`

> Destination: Singapore
> Section: Respect > Visiting homes
> 
> If invited to somebody's house, always remove your shoes before you enter as most Singaporeans do not wear their shoes at home. Socks are perfectly acceptable though. Some households may provide slippers in the bathroom, but these are generally not meant to be worn anywhere else.  
> Beware of taboos if bringing gifts. Any products (food or otherwise) involving animals may cause offence and are best avoided, as are white flowers (usually reserved for funerals). Knives and clocks are also symbols of cutting ties and death, respectively, and some Chinese are superstitious about the number four. Many Singaporean Muslims and some Hindus abstain from alcohol. Nicely packaged cookies or cakes are a safe bet, as is a bouquet of flowers from a local florist.  
> In Singapore, it is considered rude to open a gift in front of the person who gave it to you. Instead, wait until the person has left and open it in private.

### Passage 3

**Source:** [Singapore – Travel guide at Wikivoyage](https://en.wikivoyage.org/wiki/Singapore)

**Section:** Respect

**Rerank score:** -0.8207

**Found by:** semantic, keyword

**Chunk ID:** `singapore_wikivoyage-095-002-665897f09ff4`

> Destination: Singapore
> Section: Respect
> 
> Take dietary restrictions into account when inviting Singaporean friends for a meal. Many Indians and some Chinese are vegetarian. Most Malays, being Muslims, eat only halal food, while most Hindus (and a few Chinese) abstain from beef. If unsure, ask beforehand.  
> When visiting Chinese temples, do not point at the statues of deities with your index finger, as this is considered to be very rude. Use your thumb or an up-facing open palm instead. If you want to watch a street performance during the Hungry Ghost Festival, do not sit in the first row as it is traditionally reserved for the spirits of the deceased.  
> Swastikas are commonly seen in Buddhist and Hindu temples and altars. This is an ancient religious symbol that has nothing to do with Nazism.

### Passage 4

**Source:** [Singapore – Travel guide at Wikivoyage](https://en.wikivoyage.org/wiki/Singapore)

**Section:** See

**Rerank score:** -1.2709

**Found by:** semantic, keyword

**Chunk ID:** `singapore_wikivoyage-040-003-04a6a94bc954`

> Destination: Singapore
> Section: See
> 
> - Skyscrapers and shopping : The heaviest shopping mall concentration is in Orchard Road , while skyscrapers are clustered around the Singapore River , but also check out Bugis and Marina Bay to see where Singaporeans shop.  
> - Places of worship : Don't miss this aspect of Singapore, where Buddhism, Taoism, Hinduism, Sikhism, Christianity, Islam and Judaism all exist in sizeable numbers. Religious sites can be easily visited and welcome non-followers outside of service times. Particularly worth visiting include: the vast Kong Meng San Phor Kark See Monastery near Ang Mo Kio /Bishan, the colourful Hindu Sri Mariamman Temple in Chinatown , the psychedelic Burmese Buddhist Temple in Balestier and the stately Masjid Sultan in Arab Street .

## 4. Food and local experiences

**Question:** Which local dishes should I try at Singapore hawker centres?

**What to check:** Named local dishes and descriptions. Hawker-centre locations alone are insufficient.

**Review:** Pending. Mark useful / partly useful / unsupported, and explain why.

### Passage 1

**Source:** [Singapore – Travel guide at Wikivoyage](https://en.wikivoyage.org/wiki/Singapore)

**Section:** Eat > Local delicacies > Chinese cuisine

**Rerank score:** 6.1982

**Found by:** semantic, keyword

**Chunk ID:** `singapore_wikivoyage-063-000-7bee92340cc4`

> Destination: Singapore
> Section: Eat > Local delicacies > Chinese cuisine
> 
> - Bak kut teh with rice and you tiao fritters  
> - Hainanese chicken rice  
> - Fried Hokkien mee  
> - Prawn mee and pork rib soup  
> Chinese food as eaten in Singapore commonly originates from southern China , particularly Fujian , Guangdong and Hainan . While "authentic" fare is certainly available, especially in fancier restaurants, the daily fare served in hawker centres has absorbed a number of tropical touches, most notably the fairly heavy use of chilli and the Malay fermented shrimp paste belacan as condiments. Noodles can also be served not just in soup (湯 tang ), but also "dry" (干 gan ), meaning that your noodles will be served tossed with chilli and spices in one bowl, and the soup will come in a separate bowl.  
> - Bak chor mee (肉脞面) is essentially noodles with minced pork, tossed in a chilli-based sauce with lard, ikan bilis (fried anchovies), vegetables and mushrooms. Black vinegar may also be added.

### Passage 2

**Source:** [Singapore – Travel guide at Wikivoyage](https://en.wikivoyage.org/wiki/Singapore)

**Section:** Eat > Local delicacies > Peranakan/Nonya cuisine

**Rerank score:** 6.1975

**Found by:** semantic, keyword

**Chunk ID:** `singapore_wikivoyage-061-002-cbbbbcd5b305`

> Destination: Singapore
> Section: Eat > Local delicacies > Peranakan/Nonya cuisine
> 
> - Laksa , in particular the Katong laksa or laksa lemak style, is probably the best-known Singaporean dish: slippery rice noodles in a creamy, immensely rich coconut-based curry broth, topped with cockles or shrimp. The common style found in hawker centres is very spicy, although you can ask for less/no chilli to dial down the heat. The Katong style is much less spicy and is generally found only in Katong itself (see the East Coast page ). Despite sharing the same name, the dish bears almost no resemblance to the varieties found in neighbouring Malaysia.  
> - Mee siam is rice flour noodles served in a sweet-sour soup (made from tamarind, dried shrimp and fermented beans), bean curd cubes, and hard boiled eggs. Though the Chinese, Malays and Indians all have their own versions, it is the Peranakan version that is most popular with Singaporeans. You will largely find this at Malay stalls.

### Passage 3

**Source:** [Singapore – Travel guide at Wikivoyage](https://en.wikivoyage.org/wiki/Singapore)

**Section:** Eat > Local delicacies > Indian cuisine

**Rerank score:** 6.1399

**Found by:** semantic, keyword

**Chunk ID:** `singapore_wikivoyage-064-000-09961f8e8bf7`

> Destination: Singapore
> Section: Eat > Local delicacies > Indian cuisine
> 
> The smallest of Singapore's big three ethnic groups, Indians have had proportionally the smallest impact on the local culinary scene, but there is no shortage of Indian food even at many hawker centres and most neighborhoods have at least a couple of mamak (Indian Muslim) shops, often open 24 hours. Delicious and authentic Indian food can be had at Little India, including south Indian typical meals such as dosa ( thosai ) crepes, idli lentil-rice cakes and sambar soup, as well as north Indian meals including various curries, naan bread, chapati , tandoori chicken and more. In addition, however, a number of Indian dishes have been "Singaporeanised" and adopted by the entire population, including:

### Passage 4

**Source:** [Singapore – Travel guide at Wikivoyage](https://en.wikivoyage.org/wiki/Singapore)

**Section:** Eat > Hawker centres

**Rerank score:** 5.3672

**Found by:** semantic, keyword

**Chunk ID:** `singapore_wikivoyage-065-002-84346ebec82e`

> Destination: Singapore
> Section: Eat > Hawker centres
> 
> Hawker centres are built for volume and ordering from a popular hawker speaking in rapid-fire Singlish can be confusing. First stake out your territory by choping (reserve) a space by parking a friend at the table, or do what the locals do: place a packet of tissue paper on the table. If ordering from a stall not marked "self-service", note down the table number so they can deliver. The best places always have queues, so line up, and once asked place your order by stating the dish and portion size you want: "Fishball noodles, four dollars." Expect to get some or all of the following questions:  
> - "Have here or take away?" Reply "have here" to eat in, or any of "take away", "packet" or ta pao (打包) to take away.

## 5. Sample itineraries

**Question:** What could I do in the morning and evening on my first day in Singapore?

**What to check:** An itinerary with identifiable activities and their day or time-of-day context.

**Review:** Pending. Mark useful / partly useful / unsupported, and explain why.

### Passage 1

**Source:** [Explore Singapore in 4 Days: Ultimate Itinerary & Tours!](https://www.visitsingapore.com/travel-tips/travelling-to-singapore/itineraries/4-days-in-singapore/)

**Section:** Day 1: Explore the City > Evening > Singapore River

**Rerank score:** 4.6041

**Found by:** semantic, keyword

**Chunk ID:** `singapore_4_day_itinerary-007-000-5b7c20ebe831`

> Destination: Singapore
> Section: Day 1: Explore the City > Evening > Singapore River
> 
> Just before dinner, hop on a traditional bumboat by Singapore River Cruise and get to know the historic Singapore River.

### Passage 2

**Source:** [Explore Singapore in 4 Days: Ultimate Itinerary & Tours!](https://www.visitsingapore.com/travel-tips/travelling-to-singapore/itineraries/4-days-in-singapore/)

**Section:** Day 1: Explore the City > Evening > Offtrack

**Rerank score:** 3.8457

**Found by:** semantic, keyword

**Chunk ID:** `singapore_4_day_itinerary-008-000-2b3d29e840d4`

> Destination: Singapore
> Section: Day 1: Explore the City > Evening > Offtrack
> 
> Art, food and music come together at Offtrack, an effortlessly cool bar that serves up Pan Asian cuisine, cocktails and live DJ sets.

### Passage 3

**Source:** [Explore Singapore in 4 Days: Ultimate Itinerary & Tours!](https://www.visitsingapore.com/travel-tips/travelling-to-singapore/itineraries/4-days-in-singapore/)

**Section:** Day 1: Explore the City > Evening > PLUME

**Rerank score:** 3.7361

**Found by:** keyword

**Chunk ID:** `singapore_4_day_itinerary-009-000-b113caf608f3`

> Destination: Singapore
> Section: Day 1: Explore the City > Evening > PLUME
> 
> A cocktail bar inspired by the birds of Singapore, PLUME serves up intriguing cocktails and appetisers.

### Passage 4

**Source:** [Explore Singapore in 4 Days: Ultimate Itinerary & Tours!](https://www.visitsingapore.com/travel-tips/travelling-to-singapore/itineraries/4-days-in-singapore/)

**Section:** Day 1: Explore the City > Morning > Hygge

**Rerank score:** 2.9038

**Found by:** keyword

**Chunk ID:** `singapore_4_day_itinerary-003-000-61513b8da085`

> Destination: Singapore
> Section: Day 1: Explore the City > Morning > Hygge
> 
> Head to Haji Lane —a hipster haunt littered with indie outlets that stock vintage clothing and knick-knacks—such as home and lifestyle store Hygge.

## 6. Indoor activity suggestions

**Question:** What indoor attractions in Singapore can I visit on a rainy day?

**What to check:** Activities explicitly described as indoors or sheltered; do not assume a whole attraction is indoors.

**Review:** Pending. Mark useful / partly useful / unsupported, and explain why.

### Passage 1

**Source:** [Singapore – Travel guide at Wikivoyage](https://en.wikivoyage.org/wiki/Singapore)

**Section:** Do > Snow sports

**Rerank score:** 3.1805

**Found by:** semantic, keyword

**Chunk ID:** `singapore_wikivoyage-050-000-9a4c8f9f66f6`

> Destination: Singapore
> Section: Do > Snow sports
> 
> Being just one degree north of the equator, it is obviously not the best place on earth for skiing, but sunny Singapore still has a permanent indoor snow centre. Snow City offers visitors a chance to experience winter. Visitors can escape from the hot and humid tropical weather to play in snow or even learn to ski and snowboard with certified professional instructors.  
> Ice skating is also possible in Singapore, in Kallang Ice World at Leisure Park Kallang .

### Passage 2

**Source:** [Singapore – Travel guide at Wikivoyage](https://en.wikivoyage.org/wiki/Singapore)

**Section:** See

**Rerank score:** 0.9634

**Found by:** keyword

**Chunk ID:** `singapore_wikivoyage-040-001-a5b142b9fea1`

> Destination: Singapore
> Section: See
> 
> - Nature and wildlife : Popular tourist attractions in the Mandai Wildlife Reserve include the Singapore Zoo , Night Safari , Bird Paradise and River Wonders . The Mandai Boardwalk is a free scenic walk along the Upper Seletar Reservoir in the Mandai Wildlife Reserve. The Botanic Gardens are the only UNESCO World Heritage Site in Singapore. For something closer to the city, visit the futuristic Gardens by the Bay in the Marina district, behind the Marina Bay Sands. Finding "real" nature is a little harder, but the Bukit Timah Nature Reserve (in the same district as the zoo) has more plant species than in the whole of North America, and is also home to a thriving population of wild monkeys. Pulau Ubin , an island off the Changi Village in the east, is a flashback to the rural Singapore of yesteryear. Red junglefowl (often called wild chickens) can often be seen in many grassy areas around the city, even in downtown areas

### Passage 3

**Source:** [Singapore – Travel guide at Wikivoyage](https://en.wikivoyage.org/wiki/Singapore)

**Section:** Get around > On foot

**Rerank score:** 0.8678

**Found by:** semantic

**Chunk ID:** `singapore_wikivoyage-039-001-28d9cb06d00b`

> Destination: Singapore
> Section: Get around > On foot
> 
> An unavoidable downside, though, is the tropical heat and humidity , which leaves many visitors sweaty and exhausted, so do as the locals do and bring along a little towel, a bottle of water and even an umbrella to protect against the searing heat. Also, afternoon thunderstorms are fairly common during the monsoon season. It's best to get an early start, pop into air-conditioned shops, cafes and museums to cool off or take shelter from rain, and plan on heading back to the shopping mall or hotel pool before noon. Alternatively, after sundown, evenings can also be comparatively cool. On the upside, the fact that the sun is often covered in clouds and shaded by trees and greenery along roads means that you won't get as easily sunburnt as otherwise at these latitudes

### Passage 4

**Source:** [Things To Do & Must-Visit Places in Singapore](https://www.visitsingapore.com/things-to-do/top-things-to-do/)

**Section:** Curiosity Cove

**Rerank score:** 0.6004

**Found by:** semantic, keyword

**Chunk ID:** `singapore_things_to_do-017-000-4d6fadb09db3`

> Destination: Singapore
> Section: Curiosity Cove
> 
> Discover Singapore’s largest indoor nature-inspired playscape for kids 12 and under. Through interactive and educational play, children explore how animals live, adapt and thrive in the wild across four immersive zones filled with wonder and discovery.

## 7. Outdoor activity suggestions

**Question:** Where can I enjoy nature walks or cycling outdoors in Singapore?

**What to check:** Named parks, nature areas, walks, or cycling experiences.

**Review:** Pending. Mark useful / partly useful / unsupported, and explain why.

### Passage 1

**Source:** [Singapore – Travel guide at Wikivoyage](https://en.wikivoyage.org/wiki/Singapore)

**Section:** See

**Rerank score:** 2.3711

**Found by:** keyword

**Chunk ID:** `singapore_wikivoyage-040-001-a5b142b9fea1`

> Destination: Singapore
> Section: See
> 
> - Nature and wildlife : Popular tourist attractions in the Mandai Wildlife Reserve include the Singapore Zoo , Night Safari , Bird Paradise and River Wonders . The Mandai Boardwalk is a free scenic walk along the Upper Seletar Reservoir in the Mandai Wildlife Reserve. The Botanic Gardens are the only UNESCO World Heritage Site in Singapore. For something closer to the city, visit the futuristic Gardens by the Bay in the Marina district, behind the Marina Bay Sands. Finding "real" nature is a little harder, but the Bukit Timah Nature Reserve (in the same district as the zoo) has more plant species than in the whole of North America, and is also home to a thriving population of wild monkeys. Pulau Ubin , an island off the Changi Village in the east, is a flashback to the rural Singapore of yesteryear. Red junglefowl (often called wild chickens) can often be seen in many grassy areas around the city, even in downtown areas

### Passage 2

**Source:** [Things To Do & Must-Visit Places in Singapore](https://www.visitsingapore.com/things-to-do/top-things-to-do/)

**Section:** Pulau Ubin

**Rerank score:** 1.1512

**Found by:** semantic, keyword

**Chunk ID:** `singapore_things_to_do-065-000-3aff9b44c30f`

> Destination: Singapore
> Section: Pulau Ubin
> 
> Step back in time on this island with cycling trails, kayaking and a rustic charm like no other.

### Passage 3

**Source:** [Explore Singapore in 4 Days: Ultimate Itinerary & Tours!](https://www.visitsingapore.com/travel-tips/travelling-to-singapore/itineraries/4-days-in-singapore/)

**Section:** Day 3: Be One with Nature (plus a spot of shopping) > Morning > Singapore Botanic Gardens

**Rerank score:** 1.1232

**Found by:** semantic, keyword

**Chunk ID:** `singapore_4_day_itinerary-019-000-ad49a5c68d0a`

> Destination: Singapore
> Section: Day 3: Be One with Nature (plus a spot of shopping) > Morning > Singapore Botanic Gardens
> 
> Enjoy lush gardens, peace and tranquility at this 150-year-old green destination–which happens to be Singapore’s first UNESCO World Heritage Site.

### Passage 4

**Source:** [Things To Do & Must-Visit Places in Singapore](https://www.visitsingapore.com/things-to-do/top-things-to-do/)

**Section:** Exploria

**Rerank score:** 1.0601

**Found by:** keyword

**Chunk ID:** `singapore_things_to_do-023-000-d6c8b1d39bb5`

> Destination: Singapore
> Section: Exploria
> 
> Explore one of Southeast Asia's largest indoor, nature-themed multimedia attractions, where rarely seen realms of the world come to life. Use your RFID wristband to unlock interactive touchpoints and discover the wonders of nature — past and present.

## 8. Unsupported destination question

**Question:** Which ski resorts should I visit in Switzerland?

**What to check:** This Singapore knowledge base does not support Switzerland ski recommendations. Returned neighbours must not be treated as an answer.

**Review:** Pending. Mark useful / partly useful / unsupported, and explain why.

### Passage 1

**Source:** [Singapore – Travel guide at Wikivoyage](https://en.wikivoyage.org/wiki/Singapore)

**Section:** Overview

**Rerank score:** -6.0650

**Found by:** semantic, keyword

**Chunk ID:** `singapore_wikivoyage-000-001-f4220d26efdf`

> Destination: Singapore
> Section: Overview
> 
> The country has a partly deserved reputation for its sterile predictability. Nevertheless, the "Switzerland of Asia" is for many a welcome respite from the chaos, dirt and poverty of much of the rest of Southeast Asia. If you scratch below the squeaky clean surface and get away from the tourist trail you'll soon find more than meets the eye in one of the few remaining city-states in the world.

### Passage 2

**Source:** [Singapore – Travel guide at Wikivoyage](https://en.wikivoyage.org/wiki/Singapore)

**Section:** Do > Snow sports

**Rerank score:** -7.3393

**Found by:** semantic, keyword

**Chunk ID:** `singapore_wikivoyage-050-000-9a4c8f9f66f6`

> Destination: Singapore
> Section: Do > Snow sports
> 
> Being just one degree north of the equator, it is obviously not the best place on earth for skiing, but sunny Singapore still has a permanent indoor snow centre. Snow City offers visitors a chance to experience winter. Visitors can escape from the hot and humid tropical weather to play in snow or even learn to ski and snowboard with certified professional instructors.  
> Ice skating is also possible in Singapore, in Kallang Ice World at Leisure Park Kallang .

### Passage 3

**Source:** [Singapore – Travel guide at Wikivoyage](https://en.wikivoyage.org/wiki/Singapore)

**Section:** Understand > History

**Rerank score:** -10.0021

**Found by:** keyword

**Chunk ID:** `singapore_wikivoyage-006-006-52251177a10e`

> Destination: Singapore
> Section: Understand > History
> 
> In modern times, Singapore has tried to position itself as a neutral state balancing the interests of major world powers such as the United States and China. This has made Singapore a popular alternative to Switzerland for diplomatically sensitive talks between foreign leaders, such as the meeting between Chinese president Xi Jinping and Taiwanese president Ma Ying-jeou in 2015, and the meeting between American president Donald Trump and North Korean leader Kim Jong-un in 2018.

### Passage 4

**Source:** [Singapore – Travel guide at Wikivoyage](https://en.wikivoyage.org/wiki/Singapore)

**Section:** Do

**Rerank score:** -10.2534

**Found by:** semantic

**Chunk ID:** `singapore_wikivoyage-042-000-ac6fc8e69e24`

> Destination: Singapore
> Section: Do
> 
> While you can find a place to practice nearly any sport in Singapore — golfing, surfing, scuba diving, even ice skating and snow skiing — due to the country's small size your options are rather limited and prices are relatively high. For scuba diving in particular, the busy shipping lanes and sheer population pressure mean that the sea around Singapore is murky, and most locals head across the border into Malaysia (in particular Pulau Tioman or Pulau Dayang) to get their fix. On the upside, there is an abundance of dive shops in Singapore, and they often arrange weekend trips to good dive sites off the East Coast of Malaysia.

