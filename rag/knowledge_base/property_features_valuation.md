# Property Feature Valuation Guide (Indian Residential Market)

## Area (Carpet Area)
Carpet area is the single strongest predictor of residential property price in India, explaining roughly 45-50% of price variance in ML models trained on Indian housing data. Prices per square foot vary widely:
- Mumbai (South/Bandra/Powai): ₹25,000-₹80,000 per sq ft
- Bengaluru (core): ₹8,000-₹18,000 per sq ft
- Hyderabad (Gachibowli/Hitec City): ₹7,000-₹14,000 per sq ft
- Pune (Hinjewadi/Baner): ₹7,500-₹13,000 per sq ft
- Delhi NCR (Gurgaon/Noida): ₹8,000-₹22,000 per sq ft
- Chennai (OMR/Anna Nagar): ₹6,500-₹15,000 per sq ft
- Tier-2 cities: ₹3,500-₹7,000 per sq ft

## Bedrooms and Bathrooms
- Each additional bedroom typically adds 18-25% to base value for same-area property.
- Each additional bathroom adds 8-12% to the valuation.
- 3BHK units command a 10-15% per-sqft premium over 2BHK due to family demographics.
- More than 4 bathrooms has diminishing returns (luxury segment only).

## Stories / Floor Level
- Ground floor: Often discounted 5-8% due to security, privacy, and noise concerns (though preferred by elderly buyers).
- Top floor: May command premium if there is a terrace right, or discount if no lift / heat exposure.
- Middle floors (3rd-7th): Generally optimal and fetch the highest per-sqft rate.
- Multi-storey independent houses (2-3 stories): Valued 15-20% higher than flat equivalents.

## Main Road Access
Properties with main road frontage tend to command a 5-10% premium for commercial/mixed-use potential, but residential-only buyers may discount due to noise and pollution. Net effect in ML models: slightly positive.

## Guest Room
A dedicated guest room adds 5-8% to value in Indian context, particularly in tier-1 cities where hosting extended family is common.

## Basement
Basement adds 6-12% to house value, usable for storage, home office, or recreation. In waterlogging-prone areas (Mumbai, Chennai, Kolkata), basements may reduce value due to flooding risk.

## Hot Water Heating
Central hot water heating is relatively uncommon in Indian housing (most use point-of-use geysers). Properties with central hot water heating are generally premium constructions (+3-6% value).

## Air Conditioning
Pre-installed air conditioning (split AC in main rooms or central AC) adds 5-10% to price. Critical for hot climates (Delhi, Chennai, Hyderabad). Air-conditioned properties see faster transactions and lower time-on-market.

## Parking
- 0 parking spots: -5 to -10% price impact in urban markets.
- 1 covered/stilt parking: Baseline.
- 2 parking spots: +6-10% premium.
- 3+ parking spots: +12-18% (indicates luxury segment).

## Preferred Area (prefarea)
"Preferred area" in Indian real estate context refers to established, prime neighborhoods with good infrastructure, social amenities, and appreciation track record. Properties in preferred areas command 18-30% premium over comparable peripheral locations.

## Furnishing Status
- Unfurnished (furnishingstatus=0): Baseline.
- Semi-furnished (furnishingstatus=1): +3-7% (typically wardrobes, kitchen cabinets, fans/lights).
- Fully furnished (furnishingstatus=2): +8-15% (includes beds, sofa, dining, appliances). Note: resale furniture depreciates ~30% in first 2 years.

## Model Accuracy Context
The Random Forest model trained for Estate Intelligence has a reported Mean Absolute Error (MAE) of approximately ₹10,25,000. This means individual predictions can realistically vary by ±₹10 lakh from true market value. Predictions should be treated as a reference estimate, not a firm valuation. Always corroborate with:
- Comparable sold transactions in the same micro-market.
- Independent broker valuations.
- Government circle rate / ready reckoner rate.
